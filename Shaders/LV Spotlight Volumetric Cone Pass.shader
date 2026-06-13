// Made by Lereldarion (https://github.com/lereldarion/)

// Add a volumetric fog effect on VRC light volume spotlights (https://github.com/REDSIM/VRCLightVolumes)
// Visually, this shows the spotlight cones as if there was some fog.
//
// The effect is done in one screenspace drawcall. This drawcall is moderately expensive.
// In situations where many spotlights overlap, this is cheaper than the VRSL strategy of emissive drawcalls on cone mesh.
// And the FPS are more stable.
//
// Usage : configure your spotlights as VRC light volume spotlights.
// Apply the shader on a mesh that covers the entire area affected by the spot lights (an easy option is a stretched unity cube).
// Enable the *depth texture* if not already there (https://creators.vrchat.com/worlds/udon/vrc-graphics/vrc-camera-settings/).
//
// Using the internal faces of a mesh as support has 2 advantages:
// - if far from the mesh, the effect covers only part of the screen, and is less GPU intensive.
// - the mesh can use standard occlusion mecanism to be hidden entirely.
//
// The effect uses a simple fog model (Henyey-Greenstein scattering) which takes the light volume configuration as input.
// It works best if the lights are configured correctly with respect to intensity and range.
// Supported: spotlights either analytical or with cookie.
// Cookie spotlights textures should only use the 1-uv-diameter disk part of the uv square, as the corners outside of the disk are not included in the volumetric effect.
//
// Limitations:
// - The effect affects ALL light volume spotlights ! There is no nice way to only select a subset.
// - Maximum supported spotlight angle is 180 (half sphere). 
// - Not Quest compatible due to the depth reconstruction ; but the performance cost is already high on PC, so quest seems like a bad idea.
// - Spotlight with cookies: VRCLV does not enable mipmaps on the PointLightVolumeArray Texture2D ; it must be enabled manually with a script for a better effect.

// TODO list
// - fog model : add 3d noise on intensity ?
// - fog model : density gradient with distance ?

Shader "Lereldarion/LV Spotlight Volumetric Cone Pass" {
    Properties {
        _Fog_Density("Fog density", Float) = 0.1
        _Fog_Scattering_Asymmetry("Fog scattering asymmetry", Range(-1, 1)) = 0.8
        [IntRange] _Fog_Sample_Count("Fog sample count (expensive)", Range(2, 20)) = 3
        [KeywordEnum(Linear, Near Bias)] _Fog_Sampling_Distribution("Fog sample distribution", Float) = 1
        [IntRange] _Cookie_Mip("Mip for spotlight with cookie", Range(0, 16)) = 3
        [ToggleUI] _Debug_Area("Show area of effect (debug)", Float) = 0
    }
    SubShader {
        Tags {
            "Queue" = "Overlay"
            "RenderType" = "Overlay"
            "PreviewType" = "Plane"
            "IgnoreProjector" = "True"
        }
        
        // Screenspace pass
        ZWrite Off
        ZTest Always

        Cull Front // Use the backfaces of the cube, so that is shows if we are inside.
        Blend One One // Additive only

        Pass {
            CGPROGRAM
            #pragma warning (error : 3205) // implicit precision loss
            #pragma warning (error : 3206) // implicit truncation

            #pragma target 5.0
            #pragma multi_compile_instancing
            #pragma shader_feature_local _FOG_SAMPLING_DISTRIBUTION_LINEAR _FOG_SAMPLING_DISTRIBUTION_NEAR_BIAS
            
            #pragma vertex vertex_stage
            #pragma fragment fragment_stage
            
            #include "UnityCG.cginc"

            uniform float _Fog_Density;
            uniform float _Fog_Scattering_Asymmetry;
            uniform uint _Fog_Sample_Count;
            uniform float _Cookie_Mip;
            uniform float _Debug_Area;

            ///////////////////////////////////////////////////////////////////////
            // Depth reconstruction

            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);
            uniform float4 _CameraDepthTexture_TexelSize;

            // unity_MatrixInvP is not provided in BIRP. unity_CameraInvProjection is only the basic camera projection (no VR components).
            // Using d4rkpl4y3r technique of patching unity_CameraInvProjection (https://gist.github.com/d4rkc0d3r/886be3b6c233349ea6f8b4a7fcdacab3)
            float4 ClipToViewPos(float4 clipPos)
            {
                float4 normalizedClipPos = float4(clipPos.xyz / clipPos.w, 1);
                normalizedClipPos.z = 1 - normalizedClipPos.z;
                normalizedClipPos.z = normalizedClipPos.z * 2 - 1;
                float4x4 invP = unity_CameraInvProjection;
                invP._24 *= _ProjectionParams.x;
                invP._42 *= -1;
                float4 viewPos = mul(invP, normalizedClipPos);
                viewPos.y *= _ProjectionParams.x;
                return viewPos;
            }

            float3 position_vs_at_pixel(float2 pixel_position) {
                // HLSLSupport.hlsl : DepthTexture is a TextureArray in SPS-I, so its size should be safe to use to get uvs.
                float2 depth_texture_uv = pixel_position * _CameraDepthTexture_TexelSize.xy;
                float raw = SAMPLE_DEPTH_TEXTURE_LOD(_CameraDepthTexture, float4(depth_texture_uv, 0, 0)); // [0,1]

                float2 clipPos = ((pixel_position / _ScreenParams.xy) * 2 - 1) * float2(1, -1);
                #ifdef UNITY_SINGLE_PASS_STEREO
                    clipPos.x -= 2 * unity_StereoEyeIndex;
                #endif
                float4 v = ClipToViewPos(float4(clipPos, raw, 1));
                return v.xyz / v.w;
            }

            ///////////////////////////////////////////////////////////////////////
            // Geometry

            float length_sq(float3 v) { return dot(v, v); }
            static const float f32_infinity = asfloat(uint(0x7f800000));

            struct Ray {
                // Ray defined by points p(t) = origin + direction * t
                float3 origin; // also note
                float3 direction;

                void normalize() { direction = normalize(direction); }
                float3 position_at(float t) { return origin + t * direction; }

                // Compute intersection with a positive cone with `cone_origin`, cone `angle`, normalized `axis`, with a sphere cap at `radius`.
                // Angle must be less than 90 degrees. Ray must be normalized.
                // Return a pair of sorted t values that intersect on success.
                bool capped_cone_intersection(float3 cone_origin, float3 cone_axis, float cos_cone_angle, float cap_radius_sq, out float2 intersection_t);
            };

            bool Ray::capped_cone_intersection(float3 cone_origin, float3 cone_axis, float cos_cone_angle, float cap_radius_sq, out float2 intersection_t) {
                // Useful intermediates. Abbreviations : co = cone_origin, ca = cone_axis, ro = ray origin, ra = ray axis.
                const float cos_cone_angle_sq = cos_cone_angle * cos_cone_angle;
                const float3 co_ro = origin - cone_origin;
                const float dot_ca_ra = dot(cone_axis, direction);
                const float dot_ca_coro = dot(cone_axis, co_ro);
                const float dot_coro_ra = dot(co_ro, direction);
                const float dot_coro_coro = dot(co_ro, co_ro);

                // Compute intersection t candidates (><) for the ray as an infinite line  : 
                // - sphere (  ) : 0 or 2 hits
                // - symmetric infinite cone  ><  : 0 or 2 hits

                // A point p is within positive cone if: dot(ca, p-co) >= length(p-co) cos(a)
                // With ro = ray.origin, ra = normalize(ray.direction), do = ro-co :
                // dot(ca, do) + t dot(ca, ra) >= length(do + t ra) cos(a)
                // Squared: dot(ca, do)^2 + 2t dot(ca, ra) dot(ca, do) + t^2 dot(ca, ra)^2 >=
                //          cos(a)^2 (dot(do, do) + 2t dot(do, ra) + t^2 dot(ra, ra))
                // The squared version describe the symmetric cone.
                //
                // Second order equation at^2 + bt + c >= 0. Compute (a, b/2, c) in a vectorized way:
                // a = dot(ca, ra)^2 - cos(a)^2 ; dot(ra, ra) = 1
                // b / 2 =  dot(ca, ra) dot(ca, do) - cos(a)^2 dot(do, ra)
                // c = dot(ca, do)^2 - cos(a)^2 dot(do, do)
                const float3 cone_eqn_a_b2_c = float3(dot_ca_ra.xx, dot_ca_coro) * float3(dot_ca_ra, dot_ca_coro.xx) - cos_cone_angle_sq * float3(1, dot_coro_ra, dot_coro_coro);
                // delta / 4 = (b / 2)^2 - ac, solutions (-b/2 +/- sqrt(delta/4)) / a
                const float cone_eqn_delta_4 = cone_eqn_a_b2_c[1] * cone_eqn_a_b2_c[1] - cone_eqn_a_b2_c[0] * cone_eqn_a_b2_c[2];

                // Sphere intersection https://iquilezles.org/articles/intersectors/
                const float sphere_intersection_h_sq = cap_radius_sq - (dot_coro_coro - dot_coro_ra * dot_coro_ra);
                //const float sphere_intersection_h_sq = cap_radius_sq - length_sq(co_ro - dot_coro_ra * direction);

                // Early aborts scenarios :
                // - No cone hit along the line => any sphere hit cannot be within the cone
                // - No sphere hit along the line => any cone hit cannot be within the sphere
                UNITY_BRANCH if (cone_eqn_delta_4 > 0 && sphere_intersection_h_sq > 0) {
                    // Intersection raw solutions
                    const float2 cone_intersection_t = (-cone_eqn_a_b2_c[1] + float2(-1, 1) * sqrt(cone_eqn_delta_4)) / cone_eqn_a_b2_c[0];
                    const float2 sphere_intersection_t = -dot_coro_ra + float2(-1, 1) * sqrt(sphere_intersection_h_sq);
                    const float4 all_intersection_t = float4(cone_intersection_t, sphere_intersection_t);

                    // Filter solutions using the distance along the cone normal: dot(p - co, ca) = dot(ro + t ra - co, ca) = dot(ro-co, ca) + t * dot(ra, ca)
                    // Assuming spotlight angle below 180 (all on the positive plane) :
                    // Cone intersections are valid up to the sphere cap : dot distances [0, radius * cos_angle]
                    // Sphere intersections are valid for the sphere cap : dot distances >= radius * cos_angle.
                    // This is sufficient to eliminate all other unwanted intersections.
                    const float4 cone_axis_distances = dot_ca_coro + all_intersection_t * dot_ca_ra;
                    const bool4 is_positive_plane = cone_axis_distances > 0;
                    const float4 cone_axis_distances_sq = cone_axis_distances * cone_axis_distances;
                    const float sphere_cap_threshold_sq = cap_radius_sq * cos_cone_angle_sq;
                    const bool4 intersection_valid = is_positive_plane & bool4(cone_axis_distances_sq.xy <= sphere_cap_threshold_sq, cone_axis_distances_sq.zw >= sphere_cap_threshold_sq);

                    // Find at least 2 valid intersections. Less than 2 is a global miss.
                    // More than 2 is possible at edges of the cone, ignore.
                    intersection_t = intersection_valid.xy ? all_intersection_t.xy : all_intersection_t.zw;
                    bool2 valid = intersection_valid.xy | intersection_valid.zw;
                    // intersection_t contains (x|z, y|w). This covers every sets of 2 valid values except xz and yw.
                    // Try wz to reach (x|z|w, y|z|w), but only if it is not used twice (avoid returning zz or ww).
                    const bool2 pick_wz = ~valid & intersection_valid.yx;
                    intersection_t = pick_wz ? all_intersection_t.wz : intersection_t;
                    valid = pick_wz ? intersection_valid.wz : valid;
                    // Sort solutions
                    intersection_t = intersection_t.x < intersection_t.y ? intersection_t : intersection_t.yx;
                    return all(valid);
                } else {
                    intersection_t = 0; // Just to calm the compiler
                    return false;
                }
            }

            ///////////////////////////////////////////////////////////////////////
            // VRC Light Volumes https://github.com/REDSIM/VRCLightVolumes/
            
            uniform float _UdonPointLightVolumeCount; // Point Lights count, max 128
            uniform float4 _UdonPointLightVolumePosition[128]; // XYZ = Position. W = Inverse squared range (point light) | Inverse squared range, negated (spot light) | Width (area light)
            uniform float4 _UdonPointLightVolumeColor[128]; // XYZ = Color. W = Cos of angle for LUT (point light) | Cos of outer angle if no custom texture, tan of outer angle otherwise (spot light) | 2 + Height (area light) 
            uniform float4 _UdonPointLightVolumeDirection[128]; // Rotation quaternion (point light, area light, cookie spot light) | XYZ direction + W cone falloff (analytic spot light)
            uniform float3 _UdonPointLightVolumeCustomID[128]; // X = 0 if analytic, -cookie_ID, or +custom_lut_ID. Y shadow mask id. Z squared culling distance.
            uniform float _UdonPointLightVolumeCubeCount; // Cubemaps count in the custom textures array
            uniform Texture2DArray _UdonPointLightVolumeTexture; // First elements must be cubemap faces (6 face textures per cubemap). Then goes other textures
            uniform SamplerState sampler_UdonPointLightVolumeTexture;

            float3 LV_MultiplyVectorByQuaternion(float3 v, float4 q) {
                float3 t = 2.0 * cross(q.xyz, v);
                return v + q.w * t + cross(q.xyz, t);
            }
            float3 LV_PointLightAttenuation(float sqdist, float sqlightSize, float3 color, float sqMaxDist) {
                float mask = saturate(1 - sqdist / sqMaxDist);
                return mask * mask * color * sqlightSize / (sqdist + sqlightSize);
            }
            float LV_PointLightSolidAngle(float sqdist, float sqlightSize) {
                return saturate(sqrt(sqdist / (sqlightSize + sqdist)));
            }
            float LV_Smoothstep01(float x) { return x * x * (3 - 2 * x); }

            float henyey_greenstein_phase_function(float g, float cos_angle) {
                // https://en.wikipedia.org/wiki/Henyey%E2%80%93Greenstein_phase_function
                // g is the asymmetry factor, [-1, 1], usually in [0.7, 0.99] for fog
                const float g2 = g * g;
                return (1 - g2) / (4 * UNITY_PI) * pow(1 + g2 - 2 * g * cos_angle, -1.5); // div by (...)^(3/2)
            }

            void add_vrc_light_volume_light_contribution(uint light_id, Ray ray_ws, float scene_depth, inout half3 output) {
                // Based on function LV_PointLight() in LightVolumes.cginc, to understand the metadata format and use
                const float4 position = _UdonPointLightVolumePosition[light_id];
                const float4 color = _UdonPointLightVolumeColor[light_id];
                const float3 custom_id_data = _UdonPointLightVolumeCustomID[light_id];
                const int custom_id = custom_id_data.x;
                const bool is_spotlight = position.w < 0;
                
                // customId > 0 => attenuation LUT : unsupported
                // customId = 0 => parametric attenuation : supported
                // customId < 0 => parametric attenuation + cookie : supported
                UNITY_BRANCH if(!(is_spotlight && custom_id <= 0 && length_sq(color.rgb) > 0)) { return; }
                
                // Nice name to parameters
                const float4 direction_or_quaternion = _UdonPointLightVolumeDirection[light_id];
                const bool has_cookie = custom_id < 0;
                const float3 cone_origin = position.xyz;
                const float light_range_sq = custom_id_data.z; // Squared culling distance

                float3 cone_axis;
                float cos_angle;
                UNITY_BRANCH if(has_cookie) {
                    cone_axis = LV_MultiplyVectorByQuaternion(float3(0, 0, 1), float4((-1).xxx, 1) * direction_or_quaternion);
                    cos_angle = saturate(rsqrt(1 + color.w * color.w)); // color.w is tan angle
                } else {
                    cone_axis = direction_or_quaternion.xyz; // Normalized already by LV
                    cos_angle = color.w;
                }

                // Overall strategy is to compute the range of the ray within the spotlight cone, and then compute lighting based on this range.
                float2 ray_range_within_cone;
                if(ray_ws.capped_cone_intersection(position.xyz, cone_axis, cos_angle, light_range_sq, ray_range_within_cone)) {
                    ray_range_within_cone.x = max(ray_range_within_cone.x, 0); // Clamp to camera near plane
                    ray_range_within_cone.y = min(ray_range_within_cone.y, scene_depth); // Limit to scene depth
                    const float ray_range_within_cone_length = ray_range_within_cone.y - ray_range_within_cone.x;

                    UNITY_BRANCH if(ray_range_within_cone_length > 0) {
                        // Sample lighting along the ray-cone intersection length, which is split into segments (chunks).
                        // It is better to sample more close to the light (large lighting gradient) than far from it.
                        // Introduce a bias to chunk lengths : they are proportionally sized to the lerp of the light-distance of each ray intersection bound.
                        // Also provide a fallback to use purely linear sampling (equal length chunks).
                        #if defined(_FOG_SAMPLING_DISTRIBUTION_LINEAR)
                        const float2 light_distances = float2(1, 1); // Dummy equal values
                        #elif defined(_FOG_SAMPLING_DISTRIBUTION_NEAR_BIAS)
                        const float2 light_distances = dot(cone_axis, ray_ws.origin - cone_origin) + ray_range_within_cone * dot(cone_axis, ray_ws.direction);
                        #endif
                        const float2 ray_chunk_length_factors = ray_range_within_cone_length * 2 * float2(
                            light_distances.x,
                            (light_distances.y - light_distances.x) / (_Fog_Sample_Count - 1)
                        ) / (_Fog_Sample_Count * (light_distances.x + light_distances.y));
                        
                        float ray_t = ray_range_within_cone.x;
                        for(uint i = 0; i < _Fog_Sample_Count; i += 1) {
                            // Sample the next segment at the middle
                            const float ray_chunk_length = ray_chunk_length_factors.x + i * ray_chunk_length_factors.y;
                            const float3 sample_position = ray_ws.position_at(ray_t + 0.5 * ray_chunk_length);
                            ray_t += ray_chunk_length;
                            
                            // Light volume spotlight lighting code, restructured
                            float3 dir = cone_origin - sample_position;
                            float sqlen = max(dot(dir, dir), 1e-6);
                            float3 dirN = dir * rsqrt(sqlen);
                            float3 att = LV_PointLightAttenuation(sqlen, -position.w, color.rgb, light_range_sq);

                            float3 l0; // Independent of direction
                            UNITY_BRANCH if(has_cookie) {
                                // LV_SphereSpotLightCookie
                                float3 localDir = LV_MultiplyVectorByQuaternion(-dirN, direction_or_quaternion);
                                float2 uv = localDir.xy * rcp(localDir.z * color.w /* tan angle */);
                                uint id = (uint) _UdonPointLightVolumeCubeCount * 5 - custom_id - 1;
                                float3 uvid = float3(uv * 0.5 + 0.5, id);        
                                float4 cookie = _UdonPointLightVolumeTexture.SampleLevel(sampler_UdonPointLightVolumeTexture, uvid, _Cookie_Mip);
                                l0 = att * cookie.rgb * cookie.a;
                            } else {
                                // LV_SphereSpotLight
                                float spotMask = dot(cone_axis, -dirN) - cos_angle;
                                float smoothedCone = LV_Smoothstep01(saturate(spotMask * direction_or_quaternion.w));
                                l0 = att * smoothedCone;
                            }

                            // Evaluate directional lighting (l1) on dirN "normal" (main direction). Simplifies itself a lot.
                            float3 l1_on_dirN = l0 * LV_PointLightSolidAngle(sqlen, (-position.w) * saturate(1 - cos_angle));

                            float scattering = henyey_greenstein_phase_function(_Fog_Scattering_Asymmetry, dot(ray_ws.direction, dirN));
                            float3 lighting_along_ray = l0 + scattering * l1_on_dirN;

                            output += _Fog_Density * ray_chunk_length * lighting_along_ray;
                        }
                    }
                }
            }

            ///////////////////////////////////////////////////////////////////////
            // Stages

            struct VertexInput {
                float3 position_os : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct FragmentInput {
                float4 position : SV_POSITION;
                Ray ray_ws : RAY_WS;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
            };
            
            void vertex_stage (VertexInput input, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                output.position = UnityObjectToClipPos(input.position_os);

                const bool is_orthographic = UNITY_MATRIX_P._m33 == 1.0;
                const float3 camera_ws = mul(unity_MatrixInvV, float4(0, 0, 0, 1)).xyz;
                const float3 position_ws = mul(unity_ObjectToWorld, float4(input.position_os, 1)).xyz;
                if(is_orthographic) {
                    // For ray origin, project vertex position on plane normal to camera
                    output.ray_ws.direction = mul((float3x3) unity_MatrixInvV, float3(0, 0, -1));
                    output.ray_ws.origin = position_ws + output.ray_ws.direction * dot(output.ray_ws.direction, camera_ws - position_ws);
                } else {
                    output.ray_ws.direction = position_ws - camera_ws;
                    output.ray_ws.origin = camera_ws;
                }
            }

            void fragment_stage (FragmentInput input, out half4 output_color : SV_Target) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                input.ray_ws.normalize();
                
                output_color = half4(_Debug_Area.xxx * 0.5, 1);

                // Compute scene depth for ray. Used to limit cone intersection ranges.
                const float3 scene_vs = position_vs_at_pixel(input.position.xy);
                const float3 scene_ws = mul(unity_MatrixInvV, float4(scene_vs, 1)).xyz;
                const float depth = length(scene_ws - input.ray_ws.origin);
                
                const uint point_light_volume_count = min((uint) _UdonPointLightVolumeCount, 128);                
                for(uint light_id = 0; light_id < point_light_volume_count; light_id += 1) {
                    add_vrc_light_volume_light_contribution(light_id, input.ray_ws, depth, output_color.rgb);
                }
            }
            ENDCG
        }
    }
}

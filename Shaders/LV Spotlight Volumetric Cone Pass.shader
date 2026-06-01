// Made by Lereldarion (https://github.com/lereldarion/)

// Apply on a mesh that covers the entire area affected by the spot lights.
// An easy option is a stretched unity cube.

// TODO list
// - performance of tests : cone test first (0, 1, 2). camera test. if still 0 bail. On 0 ignore light
// - fog model : add transmittance to have brightness when looking at light.
// - fog model : add 3d noise on intensity ?
// - fog model : better handle intsersection range. Light falloff should be stronger
// - move camera in cone test late. If this is determinant, decrease effect ?
// - support cookie version and try to use cookie at high mipmap for light color
// - switch to depth reconstruction that works on quest

Shader "Lereldarion/LV Spotlight Volumetric Cone Pass" {
    Properties {
        _Fog_Density("Fog density", Float) = 0.1
        [ToggleUI] _Debug_Area("Debug area of effect", Float) = 0
    }
    SubShader {
        Tags {
            "Queue" = "Overlay"
            "RenderType" = "Overlay"
            "PreviewType" = "Plane"
            "IgnoreProjector" = "True"
            "DisableBatching" = "True"
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
            
            #pragma vertex vertex_stage
            #pragma fragment fragment_stage
            
            #include "UnityCG.cginc"

            uniform float _Fog_Density;
            uniform float _Debug_Area;

            ///////////////////////////////////////////////////////////////////////
            // Depth reconstruction

            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);
            uniform float4 _CameraDepthTexture_TexelSize;

            // unity_MatrixInvP is not provided in BIRP. unity_CameraInvProjection is only the basic camera projection (no VR components).
            // Using d4rkpl4y3r technique of patching unity_CameraInvProjection (https://gist.github.com/d4rkc0d3r/886be3b6c233349ea6f8b4a7fcdacab3)
            // Use after instance id have been set ! UNITY_SETUP_INSTANCE_ID(input)
            static float4x4 unity_birp_MatrixInvP;
            static float4x4 unity_birp_MatrixInvMVP;
            void setup_unity_birp_MatrixInvP() {
                float4x4 flipZ = float4x4(1, 0, 0, 0,
                                        0, 1, 0, 0,
                                        0, 0, -1, 1,
                                        0, 0, 0, 1);
                float4x4 scaleZ = float4x4(1, 0, 0, 0,
                                        0, 1, 0, 0,
                                        0, 0, 2, -1,
                                        0, 0, 0, 1);
                float4x4 invP = unity_CameraInvProjection;
                float4x4 flipY = float4x4(1, 0, 0, 0,
                                        0, _ProjectionParams.x, 0, 0,
                                        0, 0, 1, 0,
                                        0, 0, 0, 1);
                float4x4 m = mul(scaleZ, flipZ);
                m = mul(invP, m);
                m = mul(flipY, m);
                m._24 *= _ProjectionParams.x;
                m._42 *= -1;
                unity_birp_MatrixInvP = m;
                unity_birp_MatrixInvMVP = mul(unity_WorldToObject, mul(unity_MatrixInvV, unity_birp_MatrixInvP));
            }

            float3 position_vs_at_pixel(float2 pixel_position) {
                // HLSLSupport.hlsl : DepthTexture is a TextureArray in SPS-I, so its size should be safe to use to get uvs.
                float2 depth_texture_uv = pixel_position * _CameraDepthTexture_TexelSize.xy;
                float raw = SAMPLE_DEPTH_TEXTURE_LOD(_CameraDepthTexture, float4(depth_texture_uv, 0, 0)); // [0,1]
                if(!(0 < raw && raw < 1)) { discard; } // Ignore Skybox

                float2 clipPos = ((pixel_position / _ScreenParams.xy) * 2 - 1) * float2(1, -1);
                #ifdef UNITY_SINGLE_PASS_STEREO
                    clipPos.x -= 2 * unity_StereoEyeIndex;
                #endif
                float4 v = mul(unity_birp_MatrixInvP, float4(clipPos, raw, 1));
                return v.xyz / v.w;
            }

            ///////////////////////////////////////////////////////////////////////
            // Geometry

            float length_sq(float3 v) { return dot(v, v); }

            bool within_positive_cone(float3 p, float3 cone_origin, float3 cone_axis, float cos_angle_sq) {
                const float3 v = p - cone_origin;
                const float d = dot(v, cone_axis);
                if(d <= 0) { return false; }
                return d * d >= cos_angle_sq * length_sq(v);
            }
            bool within_sphere(float3 p, float3 sphere_origin, float sqr_radius) {
                return length_sq(p - sphere_origin) <= sqr_radius;
            }

            struct Ray {
                // Ray defined by points p(t) = origin + direction * t
                float3 origin; // also note
                float3 direction;

                void normalize() { direction = normalize(direction); }
                float3 position_at(float t) { return origin + t * direction; }

                // Intersection test functions. Require a normalized direction.
                // Return true and set a pair of sorted `t` intersections points if there is an intersect.
                bool cone_intersection(float3 cone_origin, float3 cone_axis, float cos_angle_sq, out float2 intersection_t);
                bool sphere_intersection(float3 sphere_center, float sphere_radius_sq, out float2 intersection_t);
            };

            bool Ray::cone_intersection(float3 cone_origin, float3 cone_axis, float cos_angle_sq, out float2 intersection_t) {
                // A point p is within positive cone if: dot(ca, p-co) >= length(p-co) cos(a)
                // With ro = ray.origin, ra = normalize(ray.direction), do = ro-co :
                // dot(ca, do) + t dot(ca, ra) >= length(do + t ra) cos(a)
                // Squared: dot(ca, do)^2 + 2t dot(ca, ra) dot(ca, do) + t^2 dot(ca, ra)^2 >=
                //          cos(a)^2 (dot(do, do) + 2t dot(do, ra) + t^2 dot(ra, ra))
                // (for simplicity we normalize ray : dot(ra, ra) = 1)
                const float3 d_origin = origin - cone_origin;
                const float dot_ca_ra = dot(cone_axis, direction);
                const float dot_ca_do = dot(cone_axis, d_origin);
                const float dot_do_ra = dot(d_origin, direction);
                const float dot_do_do = dot(d_origin, d_origin);
                // Second order equation at^2 + bt + c >= 0
                const float eqn_a = dot_ca_ra * dot_ca_ra - cos_angle_sq; // dot(ca, ra)^2 - cos(a)^2
                const float eqn_b_2 = dot_ca_ra * dot_ca_do - cos_angle_sq * dot_do_ra; // 2 (dot(ca, ra) dot(ca, do) - cos(a)^2 dot(do, ra))
                const float eqn_c = dot_ca_do * dot_ca_do - cos_angle_sq * dot_do_do; // dot(ca, do)^2 - cos(a)^2 dot(do, do)
                // delta = b^2 - 4ac, solutions (-b +/- sqrt(delta)) / 2a
                const float eqn_delta_4 = eqn_b_2 * eqn_b_2 - eqn_a * eqn_c;
                if(eqn_delta_4 <= 0) { return false; } // Ignore edge case == 0
                intersection_t = (-eqn_b_2 + float2(-1, 1) * sign(eqn_a) * sqrt(eqn_delta_4)) / eqn_a; // sign here sorts solutions in increasing order
                return true;
            }

            bool Ray::sphere_intersection(float3 sphere_center, float sphere_radius_sq, out float2 intersection_t) {
                // https://iquilezles.org/articles/intersectors/
                const float3 oc = origin - sphere_center;
                const float b = dot(oc, direction);
                const float c = dot(oc, oc) - sphere_radius_sq;
                const float h_sq = b * b - c;
                if (h_sq < 0) { return false; }
                intersection_t = -b + float2(-1, 1) * sqrt(h_sq);
                return true;
            }

            ///////////////////////////////////////////////////////////////////////
            // VRC Light Volumes https://github.com/REDSIM/VRCLightVolumes/
            
            uniform float _UdonPointLightVolumeCount; // Point Lights count, max 128
            uniform float4 _UdonPointLightVolumePosition[128]; // XYZ = Position. W = Inverse squared range (point light) | Inverse squared range, negated (spot light) | Width (area light)
            uniform float4 _UdonPointLightVolumeColor[128]; // XYZ = Color. W = Cos of angle for LUT (point light) | Cos of outer angle if no custom texture, tan of outer angle otherwise (spot light) | 2 + Height (area light) 
            uniform float4 _UdonPointLightVolumeDirection[128]; // Rotation quaternion (point light, area light, cookie spot light) | XYZ direction + W cone falloff (analytic spot light)
            uniform float3 _UdonPointLightVolumeCustomID[128]; // X = 0 if analytic, -cookie_ID, or +custom_lut_ID. Y shadow mask id. Z squared culling distance.

            float3 LV_PointLightAttenuation(float sqdist, float sqlightSize, float3 color, float sqMaxDist) {
                float mask = saturate(1 - sqdist / sqMaxDist);
                return mask * mask * color * sqlightSize / (sqdist + sqlightSize);
            }
            float LV_PointLightSolidAngle(float sqdist, float sqlightSize) {
                return saturate(sqrt(sqdist / (sqlightSize + sqdist)));
            }
            float LV_Smoothstep01(float x) { return x * x * (3 - 2 * x); }
            float LV_EvaluateSH(float L0, float3 L1, float3 n) { return L0 + dot(L1, n); }
            float3 LightVolumeEvaluate(float3 worldNormal, float3 L0, float3 L1r, float3 L1g, float3 L1b) {
                return float3(LV_EvaluateSH(L0.r, L1r, worldNormal), LV_EvaluateSH(L0.g, L1g, worldNormal), LV_EvaluateSH(L0.b, L1b, worldNormal));
            }

            void add_vrc_light_volume_light_contribution(uint light_id, Ray ray_ws, float scene_depth, inout half3 output) {
                // Based on function LV_PointLight() in LightVolumes.cginc, to understand the metadata format and use

                const float4 position = _UdonPointLightVolumePosition[light_id];
                const float4 color = _UdonPointLightVolumeColor[light_id];
                const float3 custom_id_data = _UdonPointLightVolumeCustomID[light_id];

                const bool is_spotlight = position.w < 0;
                
                // customId > 0 => attenuation LUT
                // customId = 0 => parametric attenuation
                // customId < 0 => parametric attenuation + cookie
                // Only the basic case is supported for now
                UNITY_BRANCH if(!(is_spotlight && custom_id_data.x == 0 && length_sq(color.rgb) > 0)) { return; }
                
                const float4 direction_or_rotation = _UdonPointLightVolumeDirection[light_id];

                const float3 cone_axis = direction_or_rotation.xyz;
                const float light_range_sq = custom_id_data.z; // Squared culling distance
                const float cos_angle = color.w;
                const float cos_angle_sq = cos_angle * cos_angle;

                // We want the range of the ray within the spotlight cone to sample.
                // We should have 0 <= x < y for a valid range. Start with an invalid one.
                float2 ray_range_within_cone = float2(_ProjectionParams.z /* far plane */, -1);

                // Check if camera is inside.
                if(within_sphere(ray_ws.origin, position.xyz, light_range_sq) && within_positive_cone(ray_ws.origin, position.xyz, cone_axis, cos_angle_sq)) {
                    ray_range_within_cone[0] = 0;
                }
                
                // Scan cone intersections.
                float2 cone_intersection_t;
                if(ray_ws.cone_intersection(position.xyz, cone_axis, cos_angle_sq, cone_intersection_t)) {
                    // Filter candidates : check if within the positive cone with sphere cap.
                    // t[0] <= t[1], so do not check t[0] if t[1] < 0
                    if(cone_intersection_t[1] >= 0) {
                        if(cone_intersection_t[0] >= 0) {
                            const float3 intersection = ray_ws.position_at(cone_intersection_t[0]);
                            if(dot(intersection - position.xyz, cone_axis) > 0 && within_sphere(intersection, position.xyz, light_range_sq)) {
                                ray_range_within_cone[0] = min(ray_range_within_cone[0], cone_intersection_t[0]);
                                ray_range_within_cone[1] = max(ray_range_within_cone[1], cone_intersection_t[0]);
                            }
                        }
                        {
                            const float3 intersection = ray_ws.position_at(cone_intersection_t[1]);
                            if(dot(intersection - position.xyz, cone_axis) > 0 && within_sphere(intersection, position.xyz, light_range_sq)) {
                                ray_range_within_cone[0] = min(ray_range_within_cone[0], cone_intersection_t[1]);
                                ray_range_within_cone[1] = max(ray_range_within_cone[1], cone_intersection_t[1]);
                            }
                        }
                    }
                }

                // Scan sphere caps
                float2 sphere_intersection_t;
                if(ray_ws.sphere_intersection(position.xyz, light_range_sq, sphere_intersection_t)) {
                    // Filter candidates : check if within the positive cone with sphere cap.
                    // t[0] <= t[1], so do not check t[0] if t[1] < 0
                    if(sphere_intersection_t[1] >= 0) {
                        if(sphere_intersection_t[0] >= 0 && within_positive_cone(ray_ws.position_at(sphere_intersection_t[0]), position.xyz, cone_axis, cos_angle_sq)) {
                            ray_range_within_cone[0] = min(ray_range_within_cone[0], sphere_intersection_t[0]);
                            ray_range_within_cone[1] = max(ray_range_within_cone[1], sphere_intersection_t[0]);
                        }
                        if(within_positive_cone(ray_ws.position_at(sphere_intersection_t[1]), position.xyz, cone_axis, cos_angle_sq)) {
                            ray_range_within_cone[0] = min(ray_range_within_cone[0], sphere_intersection_t[1]);
                            ray_range_within_cone[1] = max(ray_range_within_cone[1], sphere_intersection_t[1]);
                        }
                    }
                }

                // Cut range at scene_depth. If no intersection happened this is still -1.
                ray_range_within_cone[1] = min(ray_range_within_cone[1], scene_depth);
                
                if(ray_range_within_cone[1] > ray_range_within_cone[0]) {
                    // Prototype lighting
                    const float3 sample_point = ray_ws.position_at(dot(0.5, ray_range_within_cone));
                    const float lit_length = ray_range_within_cone[1] - ray_range_within_cone[0];

                    float3 dir = position.xyz - sample_point;
                    float sqlen = max(dot(dir, dir), 1e-6);
                    float3 dirN = dir * rsqrt(sqlen);
                    float spotMask = dot(direction_or_rotation.xyz, -dirN) - cos_angle;
                    float3 att = LV_PointLightAttenuation(sqlen, -position.w, color.rgb, light_range_sq);
                    float smoothedCone = LV_Smoothstep01(saturate(spotMask * direction_or_rotation.w));
                    float3 l0 = att * smoothedCone;
                    float3 l1 = dirN * LV_PointLightSolidAngle(sqlen, (-position.w) * saturate(1 - cos_angle));
                    float3 L1r = l0.r * l1;
                    float3 L1g = l0.g * l1;
                    float3 L1b = l0.b * l1;

                    // fake face at perfect angle to reflect light towards camera along its ray
                    float3 fog_normal = normalize(dirN + (-ray_ws.direction));
                    float3 sh_color = LightVolumeEvaluate(fog_normal, l0, L1r, L1g, L1b);

                    // output += _Fog_Density * color.rgb / max(color.r, max(color.g, color.b));
                    output += _Fog_Density * sh_color * lit_length;
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
                setup_unity_birp_MatrixInvP();
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

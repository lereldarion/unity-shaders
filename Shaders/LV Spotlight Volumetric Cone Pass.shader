// Made by Lereldarion (https://github.com/lereldarion/)

// Apply on a mesh that covers the entire area affected by the spot lights.
// An easy option is a stretched unity cube.

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

            ///////////////////////////////////////////////////////////////////////
            // Geometry

            float length_sq(float3 v) { return dot(v, v); }

            bool within_positive_cone(float3 p, float3 cone_origin, float3 cone_axis, float sqr_cos_angle) {
                const float3 v = p - cone_origin;
                const float d = dot(v, cone_axis);
                if(d <= 0) { return false; }
                return d * d >= sqr_cos_angle * length_sq(v);
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

                // If true, return the t of intersections between ray and the symmetric cone
                bool cone_intersection(float3 cone_origin, float3 cone_axis, float sqr_cos_angle, out float2 intersection_t);
            };

            bool Ray::cone_intersection(float3 cone_origin, float3 cone_axis, float sqr_cos_angle, out float2 intersection_t) {
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
                const float eqn_a = dot_ca_ra * dot_ca_ra - sqr_cos_angle; // dot(ca, ra)^2 - cos(a)^2
                const float eqn_b_2 = dot_ca_ra * dot_ca_do - sqr_cos_angle * dot_do_ra; // 2 (dot(ca, ra) dot(ca, do) - cos(a)^2 dot(do, ra))
                const float eqn_c = dot_ca_do * dot_ca_do - sqr_cos_angle * dot_do_do; // dot(ca, do)^2 - cos(a)^2 dot(do, do)
                // delta = b^2 - 4ac, solutions (-b +/- sqrt(delta)) / 2a
                const float eqn_delta_4 = eqn_b_2 * eqn_b_2 - eqn_a * eqn_c;
                const bool has_solutions = eqn_delta_4 > 0; // Ignore edge case == 0
                if(has_solutions) {
                    // sign here sorts solutions in increasing order
                    intersection_t = (-eqn_b_2 + float2(-1, 1) * sign(eqn_a) * sqrt(eqn_delta_4)) / eqn_a;
                }
                return has_solutions;
            }



            ///////////////////////////////////////////////////////////////////////
            // VRC Light Volumes https://github.com/REDSIM/VRCLightVolumes/
            
            uniform float _UdonPointLightVolumeCount; // Point Lights count, max 128
            uniform float4 _UdonPointLightVolumePosition[128]; // XYZ = Position. W = Inverse squared range (point light) | Inverse squared range, negated (spot light) | Width (area light)
            uniform float4 _UdonPointLightVolumeColor[128]; // XYZ = Color. W = Cos of angle for LUT (point light) | Cos of outer angle if no custom texture, tan of outer angle otherwise (spot light) | 2 + Height (area light) 
            uniform float4 _UdonPointLightVolumeDirection[128]; // Rotation quaternion (point light, area light, cookie spot light) | XYZ direction + W cone falloff (analytic spot light)
            uniform float3 _UdonPointLightVolumeCustomID[128]; // X = 0 if analytic, -cookie_ID, or +custom_lut_ID. Y shadow mask id. Z squared culling distance.

            void add_vrc_light_volume_light_contribution(uint light_id, Ray ray_ws, inout half3 output) {
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
                const float sqr_light_range = custom_id_data.z; // Squared culling distance
                const float cos_angle = color.w;
                const float sqr_cos_angle = cos_angle * cos_angle;

                // We want the range of the ray within the spotlight cone to sample.
                // We should have 0 <= x < y for a valid range. Start with an invalid one.
                float2 ray_range_within_cone = float2(_ProjectionParams.z /* far plane */, -1);

                // Check if camera is inside.
                if(within_sphere(ray_ws.origin, position.xyz, sqr_light_range) && within_positive_cone(ray_ws.origin, position.xyz, cone_axis, sqr_cos_angle)) {
                    ray_range_within_cone[0] = 0;
                }
                
                // Scan cone intersections.
                float2 cone_intersection_t = float2(-1, -1);
                if(ray_ws.cone_intersection(position.xyz, cone_axis, sqr_cos_angle, cone_intersection_t)) {
                    // Filter candidates : check if within the positive cone with sphere cap
                    if(cone_intersection_t[0] >= 0) {
                        const float3 intersection = ray_ws.position_at(cone_intersection_t[0]);
                        if(dot(intersection - position.xyz, cone_axis) > 0 && within_sphere(intersection, position.xyz, sqr_light_range)) {
                            ray_range_within_cone[0] = min(ray_range_within_cone[0], cone_intersection_t[0]);
                            ray_range_within_cone[1] = max(ray_range_within_cone[1], cone_intersection_t[0]);
                        }
                    }
                    if(cone_intersection_t[1] >= 0) {
                        const float3 intersection = ray_ws.position_at(cone_intersection_t[1]);
                        if(dot(intersection - position.xyz, cone_axis) > 0 && within_sphere(intersection, position.xyz, sqr_light_range)) {
                            ray_range_within_cone[0] = min(ray_range_within_cone[0], cone_intersection_t[1]);
                            ray_range_within_cone[1] = max(ray_range_within_cone[1], cone_intersection_t[1]);
                        }
                    }
                }
                
                if(ray_range_within_cone[1] >= ray_range_within_cone[0]) {
                    // Debug
                    float3 hue = color.rgb / max(color.r, max(color.g, color.b));
                    output += 0.2 * hue;
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
                float4 screen_position : SCREEN_POSITION;
                Ray ray_ws : RAY_WS;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
            };
            
            void vertex_stage (VertexInput input, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                output.position = UnityObjectToClipPos(input.position_os);
                output.screen_position = ComputeScreenPos(output.position);

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
                
                const uint point_light_volume_count = min((uint) _UdonPointLightVolumeCount, 128);                
                for(uint light_id = 0; light_id < point_light_volume_count; light_id += 1) {
                    add_vrc_light_volume_light_contribution(light_id, input.ray_ws, output_color.rgb);
                }
            }
            ENDCG
        }
    }
}

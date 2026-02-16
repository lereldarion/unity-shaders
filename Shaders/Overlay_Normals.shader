// Made by Lereldarion (https://github.com/lereldarion/unity-shaders)
// Free to redistribute under the MIT license

// An overlay which displays normals of triangles in world space, from data sampled in the depth texture. Requires dynamic lighting to work (for the depth texture).
//
// Initial idea from https://github.com/netri/Neitri-Unity-Shaders
// Using d4rkpl4y3r technique of patching unity_CameraInvProjection (https://gist.github.com/d4rkc0d3r/886be3b6c233349ea6f8b4a7fcdacab3)
// Improved with SPS-I support, Fullscreen "screenspace" mode, billboard sphere mode, dissolve effect

Shader "Lereldarion/Overlay/Normals" {
    Properties {
        [KeywordEnum(Mesh, Fullscreen, Billboard Sphere)] _Overlay_Mode("Overlay mode", Float) = 0
        [IntRange] _Overlay_Fullscreen_Vertex_Order("Fullscreen vertex order (mesh dependent)", Range(0, 2)) = 0
        [ToggleUI] _Overlay_Fullscreen_Only_Main_Camera("Fullscreen mode restricted to main camera", Float) = 1
        [Enum(Surface Only, 0, Filled, 1)] _Overlay_Sphere_Filled("Sphere type", Float) = 1
        [Toggle(_OVERLAY_DISSOLVE_ENABLED)] _Overlay_Dissolve("Enable radial dissolve effect", Float) = 0
        _Overlay_Noise_Texture("Noise texture for dissolve", 2D) = "" {}
        _Overlay_Radial_Dissolve("[0, 1] UV disk radial dissolve (radius start, end)", Vector) = (0.8, 1, 0, 0)
    }
    SubShader {
        Tags {
            "Queue" = "Overlay"
            "RenderType" = "Overlay"
            "VRCFallback" = "Hidden"
            "PreviewType" = "Plane"
            "IgnoreProjector" = "True"
            "DisableBatching" = "True" // For Billboard sphere mode only
        }
        
        Cull Off
        ZWrite On
        ZTest Less

        Pass {
            CGPROGRAM
            #pragma warning (error : 3205) // implicit precision loss
            #pragma warning (error : 3206) // implicit truncation

            #pragma target 5.0
            #pragma multi_compile_instancing
            #pragma multi_compile _OVERLAY_MODE_MESH _OVERLAY_MODE_FULLSCREEN _OVERLAY_MODE_BILLBOARD_SPHERE
            #pragma multi_compile __ _OVERLAY_DISSOLVE_ENABLED

            #pragma vertex vertex_stage
            #pragma fragment fragment_stage
            
            #include "UnityCG.cginc"

            struct VertexInput {
                float3 position_os : POSITION;
                float2 uv0 : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID

                #if defined(_OVERLAY_MODE_BILLBOARD_SPHERE)
                float3 normal_os : NORMAL;
                #endif
            };
            struct FragmentInput {
                sample float4 position : SV_POSITION; // Explicit interpolation modifier required here
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO

                #if defined(_OVERLAY_DISSOLVE_ENABLED)
                float2 disk_uv : DISK_UV;
                #endif

                #if defined(_OVERLAY_MODE_BILLBOARD_SPHERE)
                float3 ray_os : RAY_OS;
                float sphere_radius_os : SPHERE_RADIUS_OS;
                #endif
            };
            struct FragmentOutput {
                half4 color : SV_Target;

                #if defined(_OVERLAY_MODE_BILLBOARD_SPHERE)
                // Generate the quad in front of the sphere, and set sphere depth further away. Use conservative depth to keep early Z culling.
                // https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#ConservativeoDepth
                #if defined(UNITY_REVERSED_Z)
                float depth : SV_DepthLessEqual; // far < near (depths), [0, 1] on DX11W
                #else
                float depth : SV_DepthGreaterEqual; // near > far
                #endif
                #endif
            };

            uniform uint _Overlay_Fullscreen_Vertex_Order;
            uniform float _Overlay_Fullscreen_Only_Main_Camera;
            uniform float _Overlay_Sphere_Filled;
            uniform float2 _Overlay_Radial_Dissolve;

            UNITY_DECLARE_TEX2D(_Overlay_Noise_Texture);
            uniform float4 _Overlay_Noise_Texture_ST;

            uniform float _VRChatMirrorMode;
            uniform float _VRChatCameraMode;

            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);
            uniform float4 _CameraDepthTexture_TexelSize;

            float length_sq(float3 v) { return dot(v, v); }
            float length_sq(float2 v) { return dot(v, v); }
            static const float nan = asfloat(uint(-1)); // 0xFFF...FFF should be a quiet NaN

            // unity_MatrixInvP is not provided in BIRP. unity_CameraInvProjection is only the basic camera projection (no VR components).
            // Using d4rkpl4y3r technique of patching unity_CameraInvProjection (https://gist.github.com/d4rkc0d3r/886be3b6c233349ea6f8b4a7fcdacab3)
            float4x4 make_unity_MatrixInvP() {
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
                return m;
            }
            static float4x4 unity_MatrixInvP = make_unity_MatrixInvP();

            float3 position_vs_at_pixel(float2 pixel_position) {
                // HLSLSupport.hlsl : DepthTexture is a TextureArray in SPS-I, so its size should be safe to use to get uvs.
                float2 depth_texture_uv = pixel_position * _CameraDepthTexture_TexelSize.xy;
                float raw = SAMPLE_DEPTH_TEXTURE_LOD(_CameraDepthTexture, float4(depth_texture_uv, 0, 0)); // [0,1]

                float2 clipPos = ((pixel_position / _ScreenParams.xy) * 2 - 1) * float2(1, -1);
                #if defined(UNITY_SINGLE_PASS_STEREO)
                    clipPos.x -= 2 * unity_StereoEyeIndex;
                #endif
                float4 v = mul(unity_MatrixInvP, float4(clipPos, raw, 1));
                return v.xyz / v.w;
            }

            // https://iquilezles.org/articles/intersectors/
            float2 sphere_intersect(float3 ro, float3 rd, float4 sph) {
                const float3 oc = ro - sph.xyz;
                const float b = dot(oc, rd);
                const float c = dot(oc, oc) - sph.w * sph.w;
                float h = b * b - c;
                if(h < 0.0) { return float2(-1.0, -1.0); }
                h = sqrt(h);
                return float2(-b - h, -b + h);
            }

            bool border_pattern_discard(float2 p) {
                const float radius = length(p);

                const float transition = (radius - _Overlay_Radial_Dissolve.x) / (_Overlay_Radial_Dissolve.y - _Overlay_Radial_Dissolve.x);
                const float transition_clamped = saturate(transition);
                if(transition != transition_clamped) { return transition > 0; } // Fully kept or discarded, do not sample noise.
                const float threshold = transition_clamped * transition_clamped; // [0, 1] -> [0, 1], starts slow but fast end.

                const float2 polar = _Overlay_Noise_Texture_ST.xy * float2(atan2(p.y, p.x) / UNITY_TWO_PI, radius) +  _Time.y * _Overlay_Noise_Texture_ST.zw;
                const float2 noise_scales = float2(1, 0.5);
                const float noise = dot(noise_scales / dot(1, noise_scales), float2(
                    UNITY_SAMPLE_TEX2D_LOD(_Overlay_Noise_Texture, polar, 0).r,
                    UNITY_SAMPLE_TEX2D_LOD(_Overlay_Noise_Texture, polar * float2(-2, 2), 0).r
                ));
                return noise < threshold;
            }
            
            void vertex_stage (VertexInput input, uint vertex_id : SV_VertexID, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                float2 disk_uv = 2 * input.uv0 - 1;

                #if defined(_OVERLAY_MODE_MESH)
                const bool fullscreen = false;
                #elif defined(_OVERLAY_MODE_FULLSCREEN)
                const bool fullscreen = _VRChatMirrorMode == 0 && _VRChatCameraMode * _Overlay_Fullscreen_Only_Main_Camera == 0;
                #elif defined(_OVERLAY_MODE_BILLBOARD_SPHERE)
                const float3 plane_pos_os = input.position_os - dot(input.position_os, input.normal_os) * input.normal_os; // Assume normal_os is normalized
                const float sphere_radius_sq_os = length_sq(plane_pos_os) / max(length_sq(disk_uv), 1e-6);
                output.sphere_radius_os = sqrt(sphere_radius_sq_os);
                output.ray_os = 0;
                
                // Swap to screenspace fullscreen if within sphere, or so close to the sphere that near plane clips the surface.
                // Computing the frustum at near plane is complex. Approximate with a sphere of radius=2*near around the camera.
                const float3 camera_to_center_ws = mul(unity_ObjectToWorld, float4(0, 0, 0, 1)).xyz - _WorldSpaceCameraPos;
                const float3 biased_camera_pos_ws = _WorldSpaceCameraPos + (2 * _ProjectionParams.y) * normalize(camera_to_center_ws);
                const bool fullscreen = length_sq(mul(unity_WorldToObject, float4(biased_camera_pos_ws, 1)).xyz) <= sphere_radius_sq_os; 
                #endif

                if(fullscreen) {
                    // Fullscreen mode : cover the screen with a quad by redirecting existing vertices
                    if(vertex_id < 4) {
                        const float2 ndc = vertex_id & uint2(2, 1) ? 1 : -1; // [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                        const float2 swap = _Overlay_Fullscreen_Vertex_Order & (vertex_id & uint2(1, 2)) ? -1 : 1;
                        output.position = float4(ndc * swap, UNITY_NEAR_CLIP_VALUE, 1);
                        disk_uv = 0; // No border dissolve in fullscreen

                        #if defined(_OVERLAY_MODE_BILLBOARD_SPHERE)
                        const float4 v = mul(unity_MatrixInvP, output.position);
                        output.ray_os = mul(unity_WorldToObject, mul(unity_MatrixInvV, v.xyz / v.w));
                        #endif
                    } else {
                        output.position = nan.xxxx; // Vertex discard
                    }
                } else {
                    #if defined(_OVERLAY_MODE_BILLBOARD_SPHERE)
                    const float3 camera_pos_os = mul(unity_WorldToObject, float4(_WorldSpaceCameraPos, 1)).xyz;
                    const float camera_distance_os = length(camera_pos_os);

                    //const bool is_orthographic = UNITY_MATRIX_P._m33 == 1.0; // or OrthoParams
    
                    const float3 world_up_os = mul(unity_WorldToObject, float3(0, 1, 0));
                    const float3 billboard_normal_os = camera_pos_os / camera_distance_os;
                    const float3 billboard_x_os = normalize(cross(world_up_os, billboard_normal_os));
                    const float3 billboard_y_os = cross(billboard_normal_os, billboard_x_os);
                    const float3x3 billboard = float3x3(billboard_x_os, billboard_y_os, billboard_normal_os);
                    
                    const float apparent_radius = output.sphere_radius_os * sqrt(max(camera_distance_os - output.sphere_radius_os, 0) / (camera_distance_os + output.sphere_radius_os));
                    input.position_os = mul(float3(disk_uv * apparent_radius, output.sphere_radius_os), billboard);
                    output.ray_os = input.position_os - camera_pos_os;
                    #endif

                    output.position = UnityObjectToClipPos(input.position_os);
                }

                #if defined(_OVERLAY_DISSOLVE_ENABLED)
                output.disk_uv = disk_uv;
                #endif
            }

            void fragment_stage (FragmentInput input, out FragmentOutput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                
                #if defined(_OVERLAY_DISSOLVE_ENABLED)
                if (border_pattern_discard(input.disk_uv)) { discard; }
                #endif
                
                #if defined(_OVERLAY_MODE_BILLBOARD_SPHERE)
                const float3 camera_pos_os = mul(unity_WorldToObject, float4(_WorldSpaceCameraPos, 1)).xyz;
                const float3 ray_os = normalize(input.ray_os);
                const float2 ray_hits = sphere_intersect(camera_pos_os, ray_os, float4(0, 0, 0, input.sphere_radius_os));
                if(ray_hits.y < 0) {
                    output.depth = 0; discard; // Outside and no intersect
                } else if(ray_hits.x < 0 && _Overlay_Sphere_Filled) {
                    output.depth = UNITY_NEAR_CLIP_VALUE; // Inside sphere -> Fullscreen
                } else {
                    // Display on first sphere intersection, compute proper depth
                    const float4 sphere = UnityObjectToClipPos(camera_pos_os + (ray_hits.x >= 0 ? ray_hits.x : ray_hits.y) * ray_os);
                    output.depth = sphere.z / sphere.w;
                }
                #endif

                const float3 vs_0_0 = position_vs_at_pixel(input.position.xy);
                const float3 vs_m_0 = position_vs_at_pixel(input.position.xy + float2(-1, 0));
                const float3 vs_0_p = position_vs_at_pixel(input.position.xy + float2(0, 1));

                // Normals : cross product between pixel reconstructed VS, then WS
                const float3 normal_dir_vs = cross(vs_0_p - vs_0_0, vs_m_0 - vs_0_0);
                const float3 normal_ws = normalize(mul((float3x3) unity_MatrixInvV, normal_dir_vs));
                output.color = half4(LinearToGammaSpace(normal_ws * 0.5 + 0.5), 1);
            }
            ENDCG
        }
    }
}

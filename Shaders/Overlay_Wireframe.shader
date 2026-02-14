// Made by Lereldarion (https://github.com/lereldarion/unity-shaders)
// Free to redistribute under the MIT license

// An overlay which displays edges of triangles, from data sampled in the depth texture. Requires dynamic lighting to work (for the depth texture).
//
// Initial idea from https://github.com/netri/Neitri-Unity-Shaders
// Improved with SPS-I support, Fullscreen "screenspace" mode.
// Using d4rkpl4y3r technique of patching unity_CameraInvProjection (https://gist.github.com/d4rkc0d3r/886be3b6c233349ea6f8b4a7fcdacab3)

Shader "Lereldarion/Overlay/Wireframe" {
    Properties {
        [KeywordEnum(Mesh, Fullscreen)] _Overlay_Mode("Overlay mode", Float) = 0
        [IntRange] _Overlay_Fullscreen_Vertex_Order("Fullscreen vertex order (mesh dependent)", Range(0, 2)) = 0
        [ToggleUI] _Overlay_Fullscreen_Only_Main_Camera("Fullscreen mode restricted to main camera", Float) = 1
    }
    SubShader {
        Tags {
            "Queue" = "Overlay"
            "RenderType" = "Overlay"
            "VRCFallback" = "Hidden"
            "PreviewType" = "Plane"
            "IgnoreProjector" = "True"
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
            #pragma multi_compile _OVERLAY_MODE_MESH _OVERLAY_MODE_FULLSCREEN

            #pragma vertex vertex_stage
            #pragma fragment fragment_stage
            
            #include "UnityCG.cginc"

            struct VertexInput {
                float4 position_os : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct FragmentInput {
                float4 position : SV_POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
            };

            uniform float _Overlay_Mode;
            uniform uint _Overlay_Fullscreen_Vertex_Order;
            uniform float _Overlay_Fullscreen_Only_Main_Camera;

            uniform float _VRChatMirrorMode;
            uniform float _VRChatCameraMode;

            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);
            uniform float4 _CameraDepthTexture_TexelSize;

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
                #ifdef UNITY_SINGLE_PASS_STEREO
                    clipPos.x -= 2 * unity_StereoEyeIndex;
                #endif
                float4 v = mul(unity_MatrixInvP, float4(clipPos, raw, 1));
                return v.xyz / v.w;
            }
            
            void vertex_stage (VertexInput input, uint vertex_id : SV_VertexID, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                #if defined(_OVERLAY_MODE_MESH)
                const bool fullscreen = false;
                #elif defined(_OVERLAY_MODE_FULLSCREEN)
                const bool fullscreen = _VRChatMirrorMode == 0 && _VRChatCameraMode * _Overlay_Fullscreen_Only_Main_Camera == 0;
                #endif

                if(fullscreen) {
                    // Fullscreen mode : cover the screen with a quad by redirecting existing vertices
                    if(vertex_id < 4) {
                        const float2 ndc = vertex_id & uint2(2, 1) ? 1 : -1; // [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                        const float2 swap = _Overlay_Fullscreen_Vertex_Order & (vertex_id & uint2(1, 2)) ? -1 : 1;
                        output.position = float4(ndc * swap, UNITY_NEAR_CLIP_VALUE, 1);
                    } else {
                        output.position = nan.xxxx; // Vertex discard
                    }
                } else {
                    output.position = UnityObjectToClipPos(input.position_os);
                }
            }

            fixed4 fragment_stage (FragmentInput input) : SV_Target {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                const float3 vs_0_0 = position_vs_at_pixel(input.position.xy);
                const float3 vs_m_0 = position_vs_at_pixel(input.position.xy + float2(-1, 0));
                const float3 vs_0_p = position_vs_at_pixel(input.position.xy + float2(0, 1));
                const float3 vs_p_0 = position_vs_at_pixel(input.position.xy + float2(1, 0));
                const float3 vs_0_m = position_vs_at_pixel(input.position.xy + float2(0, -1));
                
                // 3 normals from origin, with 3 quadrants
                const float3 normal_vs_m_p = normalize(cross(vs_0_p - vs_0_0, vs_m_0 - vs_0_0));
                const float3 normal_vs_p_m = normalize(cross(vs_0_m - vs_0_0, vs_p_0 - vs_0_0));
                const float3 normal_vs_p_p = normalize(cross(vs_p_0 - vs_0_0, vs_0_p - vs_0_0));
                
                // Highlight differences in normals. Does not need WS for that.
                const float3 o = 1;
                const float sum_normal_differences = dot(o, abs(normal_vs_p_p - normal_vs_m_p)) + dot(o, abs(normal_vs_p_m - normal_vs_m_p));
                const float c = saturate(sum_normal_differences);
                return float4(c.xxx, 1);
            }
            ENDCG
        }
    }
}

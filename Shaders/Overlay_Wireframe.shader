// Made by Lereldarion (https://github.com/lereldarion/unity-shaders)
// Free to redistribute under the MIT license

// An overlay which displays edges of triangles, from data sampled in the depth texture. Requires dynamic lighting to work (for the depth texture).
//
// Initial idea from https://github.com/netri/Neitri-Unity-Shaders
// Using d4rkpl4y3r technique of patching unity_CameraInvProjection (https://gist.github.com/d4rkc0d3r/886be3b6c233349ea6f8b4a7fcdacab3)
// Improved with SPS-I support, Fullscreen "screenspace" mode, billboard sphere mode, radial dissolve effect

Shader "Lereldarion/Overlay/Wireframe" {
    Properties {
        [KeywordEnum(Mesh, Fullscreen, Billboard Sphere)] _Overlay_Mode("Overlay mode", Float) = 0
        [IntRange] _Overlay_Fullscreen_Vertex_Order("Fullscreen vertex order (mesh dependent)", Range(0, 2)) = 0
        [ToggleUI] _Overlay_Fullscreen_Only_Main_Camera("Fullscreen mode restricted to main camera", Float) = 1
        [Enum(Surface Only, 0, Filled, 1)] _Overlay_Sphere_Filled("Sphere type", Float) = 1
        [Toggle(_OVERLAY_RADIAL_DISSOLVE_ENABLED)] _Overlay_Radial_Dissolve("Enable radial dissolve effect", Float) = 0
        _Overlay_Radial_Dissolve_Noise_Texture("Noise texture for dissolve", 2D) = "" {}
        _Overlay_Radial_Dissolve_Bounds("Radial dissolve bounds (radius start, end)", Vector) = (0.8, 1, 0, 0)
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
            #pragma multi_compile __ _OVERLAY_RADIAL_DISSOLVE_ENABLED
            #pragma instancing_options procedural:vertInstancingSetup

            #pragma vertex vertex_stage
            #pragma fragment fragment_stage
            
            #include "UnityCG.cginc"
            #include "UnityStandardParticleInstancing.cginc"
            #include "common.hlsl"

            struct VertexInput {
                float3 position_os : POSITION;
                float2 uv0 : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct FragmentInput {
                sample float4 position : SV_POSITION; // Explicit interpolation modifier required here
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
                OverlayFragmentInputExtra overlay_extra;
            };
            struct FragmentOutput {
                half4 color : SV_Target;
                OverlayFragmentOutputExtra overlay_extra;
            };

            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);
            uniform float4 _CameraDepthTexture_TexelSize;

            float3 position_vs_at_pixel(float2 pixel_position) {
                // HLSLSupport.hlsl : DepthTexture is a TextureArray in SPS-I, so its size should be safe to use to get uvs.
                float2 depth_texture_uv = pixel_position * _CameraDepthTexture_TexelSize.xy;
                float raw = SAMPLE_DEPTH_TEXTURE_LOD(_CameraDepthTexture, float4(depth_texture_uv, 0, 0)); // [0,1]

                float2 clipPos = ((pixel_position / _ScreenParams.xy) * 2 - 1) * float2(1, -1);
                #ifdef UNITY_SINGLE_PASS_STEREO
                    clipPos.x -= 2 * unity_StereoEyeIndex;
                #endif
                float4 v = mul(unity_birp_MatrixInvP, float4(clipPos, raw, 1));
                return v.xyz / v.w;
            }

            void vertex_stage (VertexInput input, uint vertex_id : SV_VertexID, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                setup_unity_birp_MatrixInvP();

                output.position = OverlayObjectToClipPos(input.position_os, input.uv0, vertex_id, output.overlay_extra);
            }

            void fragment_stage (FragmentInput input, out FragmentOutput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                setup_unity_birp_MatrixInvP();

                OverlayFragment(input.overlay_extra, output.overlay_extra);

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
                output.color = half4(saturate(sum_normal_differences).xxx, 1);
            }
            ENDCG
        }
    }
}

// Made by Lereldarion (https://github.com/lereldarion/unity-shaders)
// Free to redistribute under the MIT license

// An overlay which displays a 1m grid overlapped onto world, from data sampled in the depth texture. Requires dynamic lighting to work (for the depth texture).
//
// Initial idea from https://github.com/netri/Neitri-Unity-Shaders
// Replaced emission by alpha replacement, and used nicer grid from bgolus (https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8).
// Using d4rkpl4y3r technique of patching unity_CameraInvProjection (https://gist.github.com/d4rkc0d3r/886be3b6c233349ea6f8b4a7fcdacab3)
// Improved with SPS-I support, Fullscreen "screenspace" mode, billboard sphere mode

Shader "Lereldarion/Overlay/Grid" {
    Properties {
        [Header(Grid)]
        _Grid_Size_Meters("Interval size (meters)", Float) = 1
        _Grid_Line_Width_01("Line width (% of interval)", Range(0, 1)) = 0.02

        [Header(Overlay)]
        [KeywordEnum(Mesh, Fullscreen, Billboard Sphere)] _Overlay_Mode("Overlay mode", Float) = 0
        [IntRange] _Overlay_Fullscreen_Vertex_Order("Fullscreen vertex order (mesh dependent)", Range(0, 2)) = 0
        [ToggleUI] _Overlay_Fullscreen_Only_Main_Camera("Fullscreen mode restricted to main camera", Float) = 1
        [Enum(Surface Only, 0, Filled, 1)] _Overlay_Sphere_Filled("Sphere type", Float) = 1
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
        Blend One OneMinusSrcAlpha
        ZWrite On
        ZTest Less

        Pass {
            CGPROGRAM
            #pragma warning (error : 3205) // implicit precision loss
            #pragma warning (error : 3206) // implicit truncation

            #pragma target 5.0
            #pragma multi_compile_instancing
            #pragma shader_feature_local _OVERLAY_MODE_MESH _OVERLAY_MODE_FULLSCREEN _OVERLAY_MODE_BILLBOARD_SPHERE
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
            
            uniform float _Grid_Size_Meters;
            uniform float _Grid_Line_Width_01;

            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);
            uniform float4 _CameraDepthTexture_TexelSize;

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

            void vertex_stage (VertexInput input, uint vertex_id : SV_VertexID, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                setup_unity_birp_MatrixInvP();

                output.position = OverlayObjectToClipPos(input.position_os, input.uv0, vertex_id, output.overlay_extra);
            }
            
            float3 bgolus_uv_01_grid(float3 uv, float line_width) {
                // From https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
                float3 duv = fwidth(uv);
                float3 draw_width = clamp(line_width, duv, 0.5);
                float3 grid_uv = abs(frac(uv - 0.5) * 2.0 - 1.0); // symmetrical sawtooth, 0->0, 0.5->1, 1->0
                float3 line_AA = max(duv * 1.5, 0.0000001); // Prevent div by 0 in smoothstep
                float3 pattern = smoothstep(draw_width + line_AA, draw_width - line_AA, grid_uv);
                pattern *= saturate(10 * line_width / draw_width); // fade to solid color at long distances
                pattern = lerp(pattern, line_width, saturate(duv * 2.0 - 1.0)); // fade before moire patterns
                return pattern;
            }

            void fragment_stage (FragmentInput input, out FragmentOutput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                setup_unity_birp_MatrixInvP();

                OverlayFragment(input.overlay_extra, output.overlay_extra);

                const float3 position_ws = mul(unity_MatrixInvV, float4(position_vs_at_pixel(input.position.xy), 1)).xyz;
                const float3 grid_pattern = bgolus_uv_01_grid(position_ws / _Grid_Size_Meters, _Grid_Line_Width_01);

                // Replace color only if we are on a line
                const float opacity = max(grid_pattern.x, max(grid_pattern.y, grid_pattern.z));
                output.color = half4(grid_pattern, opacity);                                
            }
            ENDCG
        }
    }
}

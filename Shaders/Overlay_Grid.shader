// Made by Lereldarion (https://github.com/lereldarion/unity-shaders)
// Free to redistribute under the MIT license

// An overlay which displays a 1m grid overlapped onto world, from data sampled in the depth texture. Requires dynamic lighting to work (for the depth texture).
//
// Initial idea from https://github.com/netri/Neitri-Unity-Shaders
// Improved with SPS-I support, Fullscreen "screenspace" mode.
// Replaced emission by alpha replacement, and used nicer grid from bgolus (https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8).
// Using d4rkpl4y3r technique of patching unity_CameraInvProjection (https://gist.github.com/d4rkc0d3r/886be3b6c233349ea6f8b4a7fcdacab3)

Shader "Lereldarion/Overlay/Grid" {
    Properties {
        [Header(Grid)]
        _Grid_Size_Meters("Interval size (meters)", Float) = 1
        _Grid_Line_Width_01("Line width (% of interval)", Range(0, 1)) = 0.02

        [Header(Overlay)]
        [ToggleUI] _Overlay_Fullscreen("Force Screenspace Fullscreen", Float) = 0
        [ToggleUI] _Overlay_Screenspace_Vertex_Reorder("Fix broken fullscreen (missing triangle due to mesh vertex order) ; mesh dependent", Float) = 0
    }
    SubShader {
        Tags {
            "Queue" = "Overlay"
            "RenderType" = "Overlay"
            "VRCFallback" = "Hidden"
            "PreviewType" = "Plane"
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
            #pragma vertex vertex_stage
            #pragma fragment fragment_stage
            #pragma multi_compile_instancing
            
            #include "UnityCG.cginc"

            struct VertexInput {
                float4 position_os : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct FragmentInput {
                float4 position : SV_POSITION; // CS as rasterizer input, screenspace as fragment input
                UNITY_VERTEX_OUTPUT_STEREO
            };
            
            uniform float _Grid_Size_Meters;
            uniform float _Grid_Line_Width_01;
            uniform float _Overlay_Fullscreen;
            uniform float _Overlay_Screenspace_Vertex_Reorder;

            uniform float _VRChatMirrorMode;
            uniform float _VRChatCameraMode;

            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);
            float4 _CameraDepthTexture_TexelSize;

            static const float nan = asfloat(uint(-1)); // 0xFFF...FFF should be a quiet NaN
            
            void vertex_stage (VertexInput input, uint vertex_id : SV_VertexID, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                if(_Overlay_Fullscreen == 1 && _VRChatMirrorMode == 0 && _VRChatCameraMode == 0) {
                    // Fullscreen mode : cover the screen with a quad by redirecting existing vertices
                    if(vertex_id < 4) {
                        float2 ndc = vertex_id & uint2(2, 1) ? 1 : -1; // [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                        if(_Overlay_Screenspace_Vertex_Reorder && (vertex_id & 1)) { ndc.x *= -1; }
                        output.position = float4(ndc, UNITY_NEAR_CLIP_VALUE, 1);
                    } else {
                        output.position = nan.xxxx; // Vertex discard
                    }
                } else {
                    output.position = UnityObjectToClipPos(input.position_os);
                }
            }

            // unity_MatrixInvP is not provided in BIRP. unity_CameraInvProjection is only the basic camera projection (no VR components).
            // Using d4rkpl4y3r technique of patching unity_CameraInvProjection (https://gist.github.com/d4rkc0d3r/886be3b6c233349ea6f8b4a7fcdacab3)
            struct DepthReconstruction {
                float2 pixel_position;
                float4x4 cs_to_vs;

                static DepthReconstruction init(float4 fragment_sv_position) {
                    DepthReconstruction o;
                    o.pixel_position = fragment_sv_position.xy;

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
                    o.cs_to_vs = mul(scaleZ, flipZ);
                    o.cs_to_vs = mul(invP, o.cs_to_vs);
                    o.cs_to_vs = mul(flipY, o.cs_to_vs);
                    o.cs_to_vs._24 *= _ProjectionParams.x;
                    o.cs_to_vs._42 *= -1;

                    return o;
                }

                float3 position_vs() {
                    return position_vs(float2(0, 0));
                }
                
                float3 position_vs(float2 pixel_shift) {
                    float2 shifted_sv_position = pixel_position + pixel_shift;
                    // HLSLSupport.hlsl : DepthTexture is a TextureArray in SPS-I, so its size should be safe to use to get uvs.
                    float2 depth_texture_uv = shifted_sv_position * _CameraDepthTexture_TexelSize.xy;
                    float raw = SAMPLE_DEPTH_TEXTURE_LOD(_CameraDepthTexture, float4(depth_texture_uv, 0, 0)); // [0,1]
                    if(!(0 < raw && raw < 1)) { discard; } // Remove skybox grid (broken)

                    float2 clipPos = ((shifted_sv_position / _ScreenParams.xy) * 2 - 1) * float2(1, -1);
                    #ifdef UNITY_SINGLE_PASS_STEREO
                        clipPos.x -= 2 * unity_StereoEyeIndex;
                    #endif
                    float4 v = mul(cs_to_vs, float4(clipPos, raw, 1));
                    return v.xyz / v.w;
                }
            };
            
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

            fixed4 fragment_stage (FragmentInput input) : SV_Target {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                DepthReconstruction dr = DepthReconstruction::init(input.position);
                float3 vs_0_0 = dr.position_vs();
                float3 ws = mul(unity_MatrixInvV, float4(vs_0_0, 1)).xyz;

                float3 grid_pattern = bgolus_uv_01_grid(ws / _Grid_Size_Meters, _Grid_Line_Width_01);

                // Replace color only if we are on a line
                float opacity = max(grid_pattern.x, max(grid_pattern.y, grid_pattern.z));
                return fixed4(grid_pattern, opacity);                                
            }
            ENDCG
        }
    }
}

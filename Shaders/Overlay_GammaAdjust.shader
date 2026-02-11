// Made by Lereldarion (https://github.com/lereldarion/unity-shaders)
// Free to redistribute under the MIT license

// An overlay which amplifies lighting by using a gamma curve with fractional exponent.
// Added fullscreen mode.

Shader "Lereldarion/Overlay/GammaAdjust" {
    Properties {
        [Header(Gamma)]
        _Gamma_Adjust_Value("Gamma Adjust Value", Range(-5, 5)) = 0
        [ToggleUI] _Transmit_Emission("Keep pixel values above 1 (emisison / bloom)", Float) = 1

        [Header(Overlay)]
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
        }
        
        Cull Off
        ZWrite On
        ZTest Less

        GrabPass { "_GammaAdjustGrabTexture" }

        Pass {
            CGPROGRAM
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
                float4 grab_screen_pos : GRAB_SCREEN_POS;
                nointerpolation float gamma : GAMMA;
                UNITY_VERTEX_OUTPUT_STEREO
            };
            
            uniform float _Gamma_Adjust_Value;
            uniform float _Transmit_Emission;

            uniform float _Overlay_Mode;
            uniform uint _Overlay_Fullscreen_Vertex_Order;
            uniform float _Overlay_Fullscreen_Only_Main_Camera;
            
            uniform float _VRChatMirrorMode;
            uniform float _VRChatCameraMode;
            
            UNITY_DECLARE_TEX2D(_GammaAdjustGrabTexture);

            static const float nan = asfloat(uint(-1)); // 0xFFF...FFF should be a quiet NaN
            
            void vertex_stage (VertexInput input, uint vertex_id : SV_VertexID, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
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
                
                output.grab_screen_pos = ComputeGrabScreenPos(output.position);
                output.gamma = exp(_Gamma_Adjust_Value); // exp(3 * (0.3 - _Gamma_Adjust_Value));
            }
            
            fixed4 fragment_stage (FragmentInput i) : SV_Target {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

                fixed4 scene_color = UNITY_SAMPLE_TEX2D_LOD(_GammaAdjustGrabTexture, i.grab_screen_pos.xy / i.grab_screen_pos.w, 0); // No mipmap as we take matching pixels
                fixed3 clamped_color = saturate(scene_color.rgb); // Avoid screen explosion at positive gamma + emission (>1) + bloom.
                fixed3 emission = _Transmit_Emission ? scene_color - clamped_color : 0;  
                return fixed4(pow(clamped_color, i.gamma) + emission, 1);
            }

            ENDCG
        }
    }
}

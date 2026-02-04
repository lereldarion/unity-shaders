// Made by Lereldarion (https://github.com/lereldarion/unity-shaders)
// Free to redistribute under the MIT license

// An overlay which amplifies lighting by using a gamma curve with fractional exponent.
// Added fullscreen mode.

Shader "Lereldarion/Overlay/GammaAdjust" {
    Properties {
        [Header(Gamma)]
        _Gamma_Adjust_Value("Gamma Adjust Value", Range(-5, 5)) = 0

        [Header(Overlay)]
        [ToggleUI] _Overlay_Fullscreen("Force Screenspace Fullscreen", Float) = 0
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
            #pragma vertex vertex_stage
            #pragma fragment fragment_stage
            #pragma multi_compile_instancing
            
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
            
            uniform float _Overlay_Fullscreen;
            uniform float _VRChatMirrorMode;
            uniform float _VRChatCameraMode;
            uniform float _Gamma_Adjust_Value;

            static const float nan = asfloat(uint(-1)); // 0xFFF...FFF should be a quiet NaN
            
            void vertex_stage (VertexInput input, uint vertex_id : SV_VertexID, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                if(_Overlay_Fullscreen == 1 && _VRChatMirrorMode == 0 && _VRChatCameraMode == 0) {
                    // Fullscreen mode : cover the screen with an oversized triangle
                    if(vertex_id < 4) {
                        // For some reason we seem to need the 4th vertex on some meshes even if the float2(3, 3) point is way out. NaN effects ?
                        float2 ndc = vertex_id & uint2(2, 1) ? 3 : -1; // [float2(-1, -1), float2(-1, 3), float2(3, -1)] to cover clip space [-1,1]^2
                        output.position = float4(ndc, UNITY_NEAR_CLIP_VALUE, 1);
                    } else {
                        output.position = nan.xxxx; // Vertex discard
                    }
                } else {
                    output.position = UnityObjectToClipPos(input.position_os);
                }
                
                output.grab_screen_pos = ComputeGrabScreenPos(output.position);
                output.gamma = exp(_Gamma_Adjust_Value); // exp(3 * (0.3 - _Gamma_Adjust_Value));
            }
            
            uniform sampler2D _GammaAdjustGrabTexture;

            fixed4 fragment_stage (FragmentInput i) : SV_Target {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

                fixed4 scene_color = tex2Dproj(_GammaAdjustGrabTexture, i.grab_screen_pos); // FIXME convert to DX11 sampler without mipmap
                fixed3 clamped_color = saturate(scene_color.rgb); // Avoid screen explosion at positive gamma + emission (>1) + bloom.
                fixed3 emission = scene_color - clamped_color;  
                return fixed4(pow(clamped_color, i.gamma) + emission, 1);
            }

            ENDCG
        }
    }
}

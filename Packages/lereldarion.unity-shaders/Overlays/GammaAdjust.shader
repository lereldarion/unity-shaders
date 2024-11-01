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
        ZWrite Off
        ZTest Less

        GrabPass { "_GammaAdjustGrabTexture" }

        Pass {
            CGPROGRAM
            #pragma target 5.0
            #pragma vertex vertex_stage
            #pragma geometry geometry_stage
            #pragma fragment fragment_stage
            #pragma multi_compile_instancing
            
            #include "UnityCG.cginc"

            struct VertexInput {
                float4 position_os : POSITION;                
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct FragmentInput {
                float4 position_cs : SV_POSITION;
                float4 grab_screen_pos : GRAB_SCREEN_POS;
                nointerpolation float gamma : GAMMA;
                UNITY_VERTEX_OUTPUT_STEREO
            };
            
            uniform float _Gamma_Adjust_Value;
            
            void vertex_stage (VertexInput input, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                output.position_cs = UnityObjectToClipPos(input.position_os);
                output.grab_screen_pos = ComputeGrabScreenPos(output.position_cs);
                output.gamma = exp(_Gamma_Adjust_Value); // exp(3 * (0.3 - _Gamma_Adjust_Value));
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
            }
            
            uniform float _Overlay_Fullscreen;
            uniform float _VRChatMirrorMode;
            uniform float _VRChatCameraMode;
            static const float4 fullscreen_quad_position_cs[4] = {
                float4(-1, -1, UNITY_NEAR_CLIP_VALUE, 1),
                float4(-1,  1, UNITY_NEAR_CLIP_VALUE, 1),
                float4( 1, -1, UNITY_NEAR_CLIP_VALUE, 1),
                float4( 1,  1, UNITY_NEAR_CLIP_VALUE, 1),
            };
            
            [maxvertexcount(4)]
            void geometry_stage(triangle FragmentInput input[3], uint triangle_id : SV_PrimitiveID, inout TriangleStream<FragmentInput> stream) {
                if(_Overlay_Fullscreen == 1 && _VRChatMirrorMode == 0 && _VRChatCameraMode == 0) {
                    // Fullscreen mode : generate a fullscreen quad for triangle 0 and discard others
                    if (triangle_id == 0) {
                        FragmentInput output = input[0];
                        UNITY_UNROLL
                        for(uint i = 0; i < 4; i += 1) {
                            output.position_cs = fullscreen_quad_position_cs[i];
                            output.grab_screen_pos = ComputeGrabScreenPos(output.position_cs);
                            stream.Append(output);
                        }
                    }
                } else {
                    // Normal geometry mode : forward triangle
                    stream.Append(input[0]);
                    stream.Append(input[1]);
                    stream.Append(input[2]);
                }
            }
            
            uniform sampler2D _GammaAdjustGrabTexture;

            fixed4 fragment_stage (FragmentInput i) : SV_Target {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

                fixed4 scene_color = tex2Dproj(_GammaAdjustGrabTexture, i.grab_screen_pos); // FIXME convert to DX11 sampler without mipmap
                return fixed4(pow(scene_color.rgb, i.gamma), 1);
            }

            ENDCG
        }
    }
}

// Made by Lereldarion (https://github.com/lereldarion/unity-shaders)
// Free to redistribute under the MIT license

// Shader with no output.
// Useful to fill an unused material slot in animations with material swaps.
// It is better to disable the renderer if possible.
Shader "Lereldarion/Hidden" {
    Properties {}
    SubShader {
        Tags {
            "RenderType" = "Opaque"
            "Queue" = "Geometry"
            "VRCFallback" = "Hidden"
            "IgnoreProjector" = "True"
        }
        
        ZTest Never
        ZWrite Off

        Pass {
            Tags {
                "LightMode" = "Always" // Used for the pass disabling trick https://github.com/d4rkc0d3r/UnityAndVRChatQuirks?tab=readme-ov-file#skipping-draw-call-for-material-slot
            }

            CGPROGRAM
            #pragma warning (error : 3205) // implicit precision loss
            #pragma warning (error : 3206) // implicit truncation
            
            #pragma target 5.0
            #pragma vertex vertex_stage
            #pragma fragment fragment_stage
            #pragma multi_compile_instancing
            #include "UnityCG.cginc"

            struct VertexInput {
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct FragmentInput {
                float4 position : SV_POSITION; // CS as rasterizer input, screenspace as fragment input
                UNITY_VERTEX_OUTPUT_STEREO
            };

            static const float nan = asfloat(uint(-1)); // 0xFFF...FFF should be a quiet NaN

            void vertex_stage (VertexInput input, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                output.position = nan.xxxx;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
            }

            fixed4 fragment_stage (FragmentInput input) : SV_Target {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                return fixed4(0, 0, 0, 1); // Should never run be required
            }
            ENDCG
        }
    }
}
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
        }
        
        ZTest Never
        ZWrite Off

        Pass {
            CGPROGRAM
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

            void vertex_stage (VertexInput input, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                output.position = float4(2., 2., 2., 1.); // Outside of clip space
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
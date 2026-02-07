// Made by Lereldarion (https://github.com/lereldarion/unity-shaders)
// Free to redistribute under the MIT license

// Overlay at optical infinity (reflecting sight), with crosshair, rangefinder distance, and worldspace compass.
// Attached to a flat surface with uniform UVs (like a quad), it will orient itself with the tangent space.
// Can be adapted to use object space if necessary.
Shader "Lereldarion/Overlay/HUD" {
    Properties {
        [Header(HUD)]
        [HDR] _Color("Emissive color", Color) = (0, 1, 0, 1)
        _Glyph_Texture_SDF ("Texture with SDF glyphs", 2D) = "white" {}
        _MSDF_Glyph_Atlas ("MSDF glyph texture", 2D) = "" {}
        _UI_Thickness ("UI thickness", Range(0, 0.003)) = 0.001
        _Test_Glyph("Glyph", Vector) = (0, 0, 1, 0)

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
        Blend OneMinusDstColor OneMinusSrcAlpha // Overlay transparency + Blacken when overlayed on white stuff
        ZWrite Off
        ZTest Less

        Pass {
            CGPROGRAM
            #pragma warning (error : 3205) // implicit precision loss
            #pragma warning (error : 3206) // implicit truncation

            #pragma target 5.0
            #pragma vertex vertex_stage
            #pragma geometry geometry_stage
            #pragma fragment fragment_stage
            #pragma multi_compile_instancing
            
            #include "UnityCG.cginc"

            uniform fixed4 _Color;
            uniform float _Overlay_Fullscreen;
            uniform float _UI_Thickness;
            uniform float4 _Test_Glyph;
            static const float _Crosshair_Circle_Radius = 0.03;
            static const float _Crosshair_Tick_Length = 0.04;
            static const float _Font_Size = 0.023;
            static const float2 _Measurement_Digit_block_Position = float2(-0.3, -0.2);

            uniform SamplerState sampler_clamp_bilinear;
            uniform Texture2D<float> _Glyph_Texture_SDF; // LEGACY
            uniform Texture2D<float3> _MSDF_Glyph_Atlas;

            uniform float _VRChatMirrorMode;
            uniform float _VRChatCameraMode;

            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);

            // Utils
            float2 pow2(float2 v) { return v * v; }
            float glsl_mod(float x, float y) { return x - y * floor(x / y); }
            static const float pi = UNITY_PI;

            // SDF anti-alias blend
            // https://blog.pkh.me/p/44-perfecting-anti-aliasing-on-signed-distance-functions.html
            // https://github.com/Chlumsky/msdfgen has other info.
            // Strategy : determine uv / sdf scale in screen space, and blend smoothly at 1px screen scale.
            // sdf should be in uv units, so both scales are equivalent. Use uv as it is continuous, sdf is not due to ifs.
            float compute_screenspace_scale_of_uv(float2 uv) {
                const float2 screenspace_uv_scales = sqrt(pow2(ddx_fine(uv)) + pow2(ddy_fine(uv)));
                return 0.5 * (screenspace_uv_scales.x + screenspace_uv_scales.y);
            }
            float sdf_blend_with_aa(float sdf, float screenspace_scale_of_uv) {
                const float w = 0.5 * screenspace_scale_of_uv;
                return smoothstep(-w, w, -sdf);
            }

            // Inigo Quilez https://iquilezles.org/articles/distfunctions2d/. Use negative for interior.
            // "psdf" = Pseudo SDF, with sharp corners. Useful to keep sharp corners when thickness is added.
            float extrude_border_with_thickness(float sdf, float thickness) {
                return abs(sdf) - thickness;
            }
            float sdf_disk(float2 p, float radius) { return length(p) - radius; }
            float sdf_circle(float2 p, float radius) { return abs(sdf_disk(p, radius)); }

            ////////////////////////////////////////////////////////////////////////////////////
            // MSDF monospace glyph rendering.
            // https://github.com/Chlumsky/msdfgen
            struct Font {
                // Atlas info from MSDF metrics. Depend on the texture used !
                static const float2 atlas_size_px = float2(256, 64);
                static const float2 cell_size_px = float2(23, 29);
                static const float atlas_distance_range_px = 2;
                static const uint grid_columns = 11;
                static const float2 glyph_bottom_left_em = float2(-0.02146, -0.07306); // metrics planeBounds, uniform due to monospace
                static const float2 glyph_size_em = float2(0.62146, 0.7452) - glyph_bottom_left_em; // metrics planeBounds, uniform due to monospace
                static const float2 cell_usable_size_px = cell_size_px - 1; // leave a half pixel on each side for padding
                static const float2 em_to_px = cell_usable_size_px / glyph_size_em;
                static const float advance_px = 0.6 * em_to_px.x;
                static const float ascender_px = 1.005 * em_to_px.y; // used for font size
                static const float line_height_px = 1.3 * em_to_px.y;
                static const float glyph_left_px = glyph_bottom_left_em.x * em_to_px.x;
                static const float2 glyph_bbox_px = float2(advance_px, cell_usable_size_px.y);

                // Atlas glyph ids
                static const uint plus = 0;
                static const uint minus = 1;
                static const uint dot = 2;
                static const uint zero = 3; // 0-9 as a sequence
                static const uint colon = 13; // :
                static const uint X = 14;
                static const uint Y = 15;
                static const uint Z = 16;
                static const uint f = 17;
                static const uint emptyset = 18;
                static const uint arrow_to = 19;
                static const uint infinity = 20;
                static const uint camera_prism = 21;
                // Atlas ids utils 
                static const uint bits = 5;
                static const uint mask = (1u << bits) - 1u;
                static const uint space = mask; // will sample to none
                static const uint packed_spaces = uint(-1);

                static float median(float3 msd) { return max(min(msd.r, msd.g), min(max(msd.r, msd.g), msd.b)); }

                // Concept : track which coordinate to sample in the texture, and only sample the chosen glyph at the end
                // Renderer "draw" methods only check if we have to draw a glyph, and register it.
                // Does not support overlap.
                float2 sampling_offset_px; // x : [0, advance_px] without left offset, y : [0, cell_usable_size_px.y]
                float sampling_inverse_scale;
                uint sampling_atlas_id;

                static Font init() {
                    Font r;
                    r.sampling_offset_px = float2(0, 0);
                    r.sampling_inverse_scale = 1;
                    r.sampling_atlas_id = space;
                    return r;
                }
                float sdf() {
                    const uint atlas_row = sampling_atlas_id / grid_columns;
                    const uint atlas_column = sampling_atlas_id - atlas_row * grid_columns;
                    const float2 atlas_offset_px = float2(atlas_column * cell_size_px.x, atlas_size_px.y - cell_size_px.y * (atlas_row + 1)) + 0.5;
                    const float2 glyph_offset_px = sampling_offset_px - float2(glyph_left_px, 0);
                    const float tex_sd = median(_MSDF_Glyph_Atlas.SampleLevel(sampler_clamp_bilinear, (glyph_offset_px + atlas_offset_px) / atlas_size_px, 0)) - 0.5;
                    // tex_sd is in [-0.5, 0.5]. It represents texture pixel ranges between [-msdf_pixel_range, msdf_pixel_range], using the inverse SDF direction.
                    const float tex_sd_pixel = -tex_sd * 2 * atlas_distance_range_px;
                    return sampling_inverse_scale * tex_sd_pixel;
                }
                void draw_glyph(float2 p, uint glyph, float2 position, float size) {
                    const float scale = ascender_px / size;
                    const float2 px = (p - position) * scale;
                    if(all(0 <= px && px <= glyph_bbox_px)) {
                        sampling_offset_px = px;
                        sampling_inverse_scale = 1 / scale;
                        sampling_atlas_id = glyph;
                    }
                }
                void draw_glyphs_6x12(float2 p, uint4 glyphs[3], float2 position, float size) {
                    // glyphs : X/Z=00fedcba, Y/W=00lkjihg
                    const float scale = ascender_px / size;
                    const float2 px = (p - position) * scale;
                    if(all(0 <= px && px <= glyph_bbox_px * float2(12, 6))) {
                        float2 column_row = floor(px / glyph_bbox_px);
                        sampling_offset_px = px - column_row * glyph_bbox_px;
                        sampling_inverse_scale = 1 / scale;

                        uint n = column_row.x;
                        uint row = 5 - column_row.y;
                        uint2 glyphs_12 = row < 3 ? glyphs[row].xy : glyphs[row - 3].zw;
                        uint glyphs_6 = n < 6 ? glyphs_12.x : glyphs_12.y;
                        n = n >= 6 ? n - 6 : n;
                        sampling_atlas_id = (glyphs_6 >> (n * Font::bits)) & Font::mask;
                    }
                }
            };

            ////////////////////////////////////////////////////////////////////////////////////

            // Data that can be precomputed at vertex stage.
            struct HudData {
                // Unit vectors that stay aligned to the vertical direction
                nointerpolation float3 aligned_uv_x_ws : ALIGNED_UV_X;
                nointerpolation float3 aligned_uv_y_ws : ALIGNED_UV_Y;

                // "X -123456.89" : 12 glyphs, 5 bits each. 6 per u32. 
                nointerpolation uint4 glyphs[3] : GLYPHS; // X/Z=00fedcba, Y/W=00lkjihg, [i].XY = world pos, [i].ZW = {far plane, range, fps}

                nointerpolation float azimuth_radiants : AZIMUTH;
                nointerpolation float elevation_radiants : ELEVATION;

                static HudData compute(float3 normal_ws, float3 tangent_ws, bool normal_is_forward) {
                    HudData hd;

                    float forward_flip = normal_is_forward ? 1 : -1;
                    float3 forward_normal_ws = normal_ws * forward_flip;
                    float3 forward_tangent_ws = tangent_ws * forward_flip;

                    // Similar skybox coordinate system but y stays aligned to worldspace vertical.
                    const float3 up_direction_ws = float3(0, 1, 0);
                    const float3 horizontal_tangent = normalize(cross(forward_normal_ws, up_direction_ws));
                    hd.aligned_uv_x_ws = horizontal_tangent * -1 /* needed but not sure why ; ht should go to the right */;
                    hd.aligned_uv_y_ws = cross(horizontal_tangent, forward_normal_ws);

                    // World azimuth and elevation of the surface forward normal
                    const float3 east_ws = float3(1, 0, 0);
                    const float3 north_ws = float3(0, 0, 1);
                    const float angular_dist_to_north_0_pi = acos(dot(horizontal_tangent, east_ws));
                    hd.azimuth_radiants = pi - angular_dist_to_north_0_pi * sign(dot(horizontal_tangent, north_ws)); // 0 at north, pi/2 east, pi south, 3pi/2 west
                    const float angular_dist_to_up_0_pi = acos(dot(normal_ws, up_direction_ws));
                    hd.elevation_radiants = pi/2 - angular_dist_to_up_0_pi; // -pi/2 when looking at the bottom, pi/2 at the top

                    // Compute depth from the depth texture.
                    // Sample at the crosshair center, which means aligned with the normal of the quad.
                    // Always use data from the first eye to have matching ranges between eyes.
                    #if UNITY_SINGLE_PASS_STEREO
                    const float3 camera_pos_ws = unity_StereoWorldSpaceCameraPos[0];
                    const float4x4 matrix_vp = unity_StereoMatrixVP[0];
                    #else
                    const float3 camera_pos_ws = _WorldSpaceCameraPos;
                    const float4x4 matrix_vp = UNITY_MATRIX_VP;
                    #endif
                    const float3 sample_point_ws = camera_pos_ws + forward_normal_ws;
                    const float4 sample_point_cs = mul(matrix_vp, float4(sample_point_ws, 1)); // UnityWorldToClipPos()
                    float4 screen_pos = ComputeNonStereoScreenPos(sample_point_cs);
                    #if UNITY_SINGLE_PASS_STEREO
                    // o.xy = TransformStereoScreenSpaceTex(o.xy, pos.w);
                    screen_pos.xy = screen_pos.xy * unity_StereoScaleOffset[0].xy + unity_StereoScaleOffset[0].zw * screen_pos.w;
                    #endif
                    const float depth_texture_value = SAMPLE_DEPTH_TEXTURE_LOD(_CameraDepthTexture, float4(screen_pos.xy / screen_pos.w, 0, 4 /* mipmap level = average a little bit */));
                    const float range_ws = LinearEyeDepth(depth_texture_value) / sample_point_cs.w;

                    // Pre compute position glyphs. Simplifies the fragment code.
                    const uint bits = Font::bits; // 5 bits, 6 glyphs bit packed per uint
                    {
                        // World pos (camera). May be negative
                        uint3 pad = camera_pos_ws < 0 ? Font::minus : Font::space; // Start with a '-' if needed. After being used once, reset to space.
                        uint3 v = uint3(abs(camera_pos_ws) * 100);
                        uint3 glyphs[2];
                        glyphs[1]  =                Font::zero + v % 10;                                   glyphs[1] <<= bits; v /= 10;
                        glyphs[1] |=                Font::zero + v % 10;                                   glyphs[1] <<= bits; v /= 10;
                        glyphs[1] |=                          Font::dot;                                   glyphs[1] <<= bits;
                        glyphs[1] |=                Font::zero + v % 10;                                   glyphs[1] <<= bits; v /= 10;
                        glyphs[1] |= v == 0 ? pad : Font::zero + v % 10; pad = v == 0 ? Font::space : pad; glyphs[1] <<= bits; v /= 10;
                        glyphs[1] |= v == 0 ? pad : Font::zero + v % 10; pad = v == 0 ? Font::space : pad;                     v /= 10;
                        glyphs[0]  = v == 0 ? pad : Font::zero + v % 10; pad = v == 0 ? Font::space : pad; glyphs[0] <<= bits; v /= 10;
                        glyphs[0] |= v == 0 ? pad : Font::zero + v % 10; pad = v == 0 ? Font::space : pad; glyphs[0] <<= bits; v /= 10;
                        glyphs[0] |= v == 0 ? pad : Font::zero + v % 10; pad = v == 0 ? Font::space : pad; glyphs[0] <<= bits; v /= 10;
                        glyphs[0] |= v == 0 ? pad : Font::zero + v % 10; pad = v == 0 ? Font::space : pad; glyphs[0] <<= bits; v /= 10;
                        glyphs[0] |=                                pad;                                   glyphs[0] <<= bits;
                        // If v > 0 at this stage, too large to be represented, use infinity symbol. Unlikely to happen in VRC.
                        glyphs[1] = v == 0 ? glyphs[1] : (Font::packed_spaces >> (32 - 4 * bits)) | ((camera_pos_ws < 0 ? Font::minus : Font::space) << (4 * bits)) | (Font::infinity << (5 * bits));
                        glyphs[0] = v == 0 ? glyphs[0] : Font::packed_spaces << bits;
                        // Finalize with field symbol and repack
                        glyphs[0] |= uint3(Font::X, Font::Y, Font::Z);
                        hd.glyphs[0].xy = uint2(glyphs[0].x, glyphs[1].x);
                        hd.glyphs[1].xy = uint2(glyphs[0].y, glyphs[1].y);
                        hd.glyphs[2].xy = uint2(glyphs[0].z, glyphs[1].z);
                    }
                    {
                        // Far plane, Range, Fps (1/smoothDt). Never negative.
                        uint3 v = uint3(max(float3(_ProjectionParams.z, range_ws, unity_DeltaTime.w), 0) * 100);
                        uint3 glyphs[2];
                        glyphs[1]  =                        Font::zero + v % 10; glyphs[1] <<= bits; v /= 10;
                        glyphs[1] |=                        Font::zero + v % 10; glyphs[1] <<= bits; v /= 10;
                        glyphs[1] |=                                  Font::dot; glyphs[1] <<= bits;
                        glyphs[1] |=                        Font::zero + v % 10; glyphs[1] <<= bits; v /= 10;
                        glyphs[1] |= v == 0 ? Font::space : Font::zero + v % 10; glyphs[1] <<= bits; v /= 10;
                        glyphs[1] |= v == 0 ? Font::space : Font::zero + v % 10;                     v /= 10;
                        glyphs[0]  = v == 0 ? Font::space : Font::zero + v % 10; glyphs[0] <<= bits; v /= 10;
                        glyphs[0] |= v == 0 ? Font::space : Font::zero + v % 10; glyphs[0] <<= bits; v /= 10;
                        glyphs[0] |= v == 0 ? Font::space : Font::zero + v % 10; glyphs[0] <<= bits; v /= 10;
                        glyphs[0] |= v == 0 ? Font::space : Font::zero + v % 10; glyphs[0] <<= bits; v /= 10;
                        glyphs[0] |=                                Font::space; glyphs[0] <<= bits;
                        // If v > 0 at this stage, too large to be represented, use infinity symbol.
                        glyphs[1] = v == 0 ? glyphs[1] : (Font::packed_spaces >> (32 - 5 * bits)) | (Font::infinity << (5 * bits));
                        glyphs[0] = v == 0 ? glyphs[0] : Font::packed_spaces << bits;
                        // Range undefined == 0. Signal with emptyset
                        if(range_ws < 0.001) {
                            glyphs[1].y = (Font::packed_spaces >> (32 - 5 * bits)) | (Font::emptyset << (5 * bits));
                            glyphs[0].y = Font::packed_spaces << bits;
                        }
                        // Finalize with field symbol and repack
                        glyphs[0] |= uint3(Font::camera_prism, Font::arrow_to, Font::f);
                        hd.glyphs[0].zw = uint2(glyphs[0].x, glyphs[1].x);
                        hd.glyphs[1].zw = uint2(glyphs[0].y, glyphs[1].y);
                        hd.glyphs[2].zw = uint2(glyphs[0].z, glyphs[1].z);
                    }
                    return hd;
                }
            };

            ////////////////////////////////////////////////////////////////////////////////////

            struct VertexInput {
                float4 position_os : POSITION;
                float3 normal_os : NORMAL;
                float4 tangent_os : TANGENT;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct FragmentInput {
                float4 position : SV_POSITION;
                float3 camera_to_geometry_ws : CAMERA_TO_GEOMETRY_WS;
                HudData hud_data;

                UNITY_VERTEX_OUTPUT_STEREO
            };

            void vertex_stage (VertexInput input, out VertexInput output) {
                output = input;
            }

            ////////////////////////////////////////////////////////////////////////////////////

            [maxvertexcount(4)]
            void geometry_stage(triangle VertexInput input[3], uint triangle_id : SV_PrimitiveID, inout TriangleStream<FragmentInput> stream) {
                UNITY_SETUP_INSTANCE_ID(input[0]);
                
                FragmentInput output;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                UNITY_BRANCH
                if(_Overlay_Fullscreen == 1 && _VRChatMirrorMode == 0 && _VRChatCameraMode == 0) {
                    // Fullscreen mode : generate a fullscreen quad for triangle 0 and discard others
                    if (triangle_id == 0) {
                        float3 normal_ws = normalize(mul((float3x3) unity_MatrixInvV, float3(0, 0, -1)));
                        float3 tangent_ws = normalize(mul((float3x3) unity_MatrixInvV, float3(1, 0, 0)));
                        output.hud_data = HudData::compute(normal_ws, tangent_ws, true);

                        // Generate in VS close to near clip plane. Having non CS positions is essential to return to WS later.
                        float2 quad[4] = { float2(-1, -1), float2(-1, 1), float2(1, -1), float2(1, 1) };
                        float near_plane_z = -_ProjectionParams.y;
                        float2 tan_half_fov = 1 / unity_CameraProjection._m00_m11; // https://jsantell.com/3d-projection/
                        // Add margins in case the matrix has some rotation/skew
                        float quad_z = near_plane_z * 2; // z margin
                        float quad_xy = quad_z * tan_half_fov * 1.2; // xy margin

                        UNITY_UNROLL
                        for(uint i = 0; i < 4; i += 1) {
                            float4 position_vs = float4(quad[i] * quad_xy, quad_z, 1);
                            output.position = UnityViewToClipPos(position_vs);
                            output.camera_to_geometry_ws = mul((float3x3) unity_MatrixInvV, position_vs.xyz);
                            stream.Append(output);
                        }
                    }
                } else {
                    // Normal geometry mode : forward triangle

                    // Use TBN from vertex 0 to generate data.
                    // TBN should be uniform on the mesh for the shader to work anyway.
                    // Using WS TBN is no more costly than OS, because OS may be skewed by skinning in a skinned mesh.
                    float3 normal_ws = UnityObjectToWorldNormal(input[0].normal_os);
                    float3 tangent_ws = UnityObjectToWorldNormal(input[0].tangent_os.xyz);
                    
                    // Flip TBN to have forward orientation. This make the HUD 2 sided with mirror.
                    float4 triangle_barycenter_os = (1./3.) * (input[0].position_os + input[1].position_os + input[2].position_os);
                    float3 camera_to_triangle_ws = mul(unity_ObjectToWorld, triangle_barycenter_os).xyz - _WorldSpaceCameraPos;
                    bool normal_is_forward = dot(normal_ws, camera_to_triangle_ws) >= 0;
                    output.hud_data = HudData::compute(normal_ws, tangent_ws, normal_is_forward);
                    
                    UNITY_UNROLL
                    for(uint i = 0; i < 3; i += 1) {
                        output.position = UnityObjectToClipPos(input[i].position_os);
                        output.camera_to_geometry_ws = mul(unity_ObjectToWorld, input[i].position_os).xyz - _WorldSpaceCameraPos;
                        stream.Append(output);
                    }
                }
            }

            ////////////////////////////////////////////////////////////////////////////////////
            // Glyph SDF system.
            
            // SDF texture containing glyphs, and metadata for each glyph.
            // Texture is generated using TextMeshPro and extracted afterwards (using screenshot of preview, as "Extract Atlas" option did not work).
            // Metadata copied by hand for now.
            struct GlyphDefinition {
                // https://learnopengl.com/In-Practice/Text-Rendering
                // all sizes in px with respect to the texture
                float2 offset;
                float2 size;
                // offset of top left corner in px when placing the rect
                float2 horizontal_bearing;
                // offset for the next character origin
                float advance;
            };
            static const GlyphDefinition glyph_definition_table[13] = {
                { float2( 10,  10), float2(44, 64), float2(3.51, 62.84 - 64), 50.0 }, // 0
                { float2(135, 173), float2(40, 62), float2(6.85, 61.92 - 62), 50.0 }, // 1
                { float2(138,  10), float2(42, 63), float2(4.53, 62.84 - 63), 50.0 }, // 2
                { float2( 10,  93), float2(44, 64), float2(3.42, 62.84 - 64), 50.0 }, // 3
                { float2( 73,  10), float2(46, 62), float2(2.06, 61.92 - 62), 50.0 }, // 4
                { float2( 73,  91), float2(44, 63), float2(3.60, 61.92 - 63), 50.0 }, // 5
                { float2( 73, 173), float2(43, 64), float2(4.58, 62.84 - 64), 50.0 }, // 6
                { float2(136,  92), float2(42, 62), float2(4.61, 61.92 - 62), 50.0 }, // 7
                { float2( 10, 176), float2(44, 64), float2(3.91, 62.84 - 64), 50.0 }, // 8
                { float2(199,  10), float2(42, 64), float2(4.22, 62.84 - 64), 50.0 }, // 9
                { float2(197,  93), float2(45, 45), float2(4.39, 51.94 - 45), 52.6 }, // +
                { float2(197, 157), float2(22,  8), float2(4.00, 27.42 -  8), 30.0 }, // -
                { float2(194, 184), float2( 9, 10), float2(8.22,  9.63 - 10), 25.0 }, // .
            };
            static const float glyph_texture_resolution = 256; // resolution at which definition values have been computed
            static const float2 glyph_max_box_size = float2(50.1, 70); // Used for UI spacing, chosen by hand from above data
            

            // A glyph renderer checks if each added character bounds contain the current pixel, and updates glyph texture uv when it does.
            // At the end we can sample only once the texture to get the SDF value.
            // This value is the value of the last character touching the current pixel for this renderer.
            // Pros : only one texture sample.
            // Cons : no overlap between characters of a renderer (but you can have overlaps by merging SDFs from 2 renderers).
            struct GlyphRendererOld {
                // Accumulator : which pixels to sample in the glyph table for the current pixel
                float2 glyph_texture_coord;

                void add(GlyphDefinition glyph, float2 pixel_uv, float2 origin_uv, float scale) {
                    float2 glyph_drawing_space = (pixel_uv - origin_uv) / scale;
                    float2 glyph_box_coord = glyph_drawing_space - glyph.horizontal_bearing;
                    bool within_glyph_box = all(0 <= glyph_box_coord && glyph_box_coord <= glyph.size);
                    if(within_glyph_box) {
                        glyph_texture_coord = glyph_box_coord + glyph.offset;
                    }
                }

                float2 add_left(uint glyph_id, float2 pixel_uv, float2 origin_uv, float scale) {
                    GlyphDefinition glyph = glyph_definition_table[glyph_id];
                    origin_uv = origin_uv - float2(glyph.advance * scale, 0);
                    add(glyph, pixel_uv, origin_uv, scale);
                    return origin_uv;
                }

                float2 add_right(uint glyph_id, float2 pixel_uv, float2 origin_uv, float scale) {
                    GlyphDefinition glyph = glyph_definition_table[glyph_id];
                    add(glyph, pixel_uv, origin_uv, scale);
                    return origin_uv + float2(glyph.advance * scale, 0);
                }

                float sdf(float thickness) {
                    const float2 glyph_texture_uv = glyph_texture_coord / glyph_texture_resolution;
                    // Force mipmap 0, as we have artefacts with auto mipmap (derivatives are propably noisy). Texture is small anyway.
                    const float tex_sdf = _Glyph_Texture_SDF.SampleLevel(sampler_clamp_bilinear, glyph_texture_uv, 0);
                    // 1 interior, 0 exterior
                    return (1 - tex_sdf) - thickness;
                }

                static GlyphRendererOld create() {
                    GlyphRendererOld r;
                    // Usually the corners are outside glyphs
                    r.glyph_texture_coord = float2(0, 0);
                    return r;
                }
            };

            ////////////////////////////////////////////////////////////////////////////////////
            // Sight

            float sdf_crosshair_0thickness(float2 p) {
                float circle = abs(sdf_disk(p, _Crosshair_Circle_Radius));

                // 4 symmetric segments. Use symmetry to only look at east one.
                p = abs(p);
                p = p.x > p.y ? p : p.yx;
                float2 closest_segment_point = float2(clamp(p.x, _Crosshair_Circle_Radius - 0.5 * _Crosshair_Tick_Length, _Crosshair_Circle_Radius + 0.5 * _Crosshair_Tick_Length), 0);
                float cross = distance(p, closest_segment_point);
                return min(circle, cross);
            }

            ////////////////////////////////////////////////////////////////////////////////////
            // Compass

            float3 closest_step_positioning(float angle_rad, float step_interval_deg) {
                const float step_interval_rad = step_interval_deg / 180. * pi;
                const float unit_coordinate = angle_rad / step_interval_rad;
                // Positionning in "unit" coordinates
                const float closest_step = round(unit_coordinate);
                const float angle_difference_to_closest_step = unit_coordinate - closest_step;
                const float distance_to_closest_step = abs(angle_difference_to_closest_step); // [-0.5, 0.5]
                return float3(closest_step, angle_difference_to_closest_step, distance_to_closest_step) * step_interval_rad;
            }

            float interval_1d_centered_sdf(float x, float center, float radius) {
                return abs(x - center) - radius;
            }

            float interval_1d_sdf(float x, float from, float to) {
                return interval_1d_centered_sdf(x, (from + to) / 2., (to - from) / 2.);
            }

            float elevation_display_sdf(float2 uv, float elevation_at_0, inout GlyphRendererOld renderer) {
                const float tick_start_x = 0.2;
                const float tick_length = 0.01;
                const float legend_start_x = tick_start_x + tick_length * 2.3;
                const float glyph_scale = 0.0003;
                
                // UVs are sin angle from normal ~ angle for center, in radiants
                const float pixel_elevation = elevation_at_0 + uv.y;

                // Tick marks : vertical column & distance to nearest tick using round(y)
                const float3 ticks_1_deg = closest_step_positioning(pixel_elevation, 1.);
                const float3 ticks_10_deg = closest_step_positioning(pixel_elevation, 10.);
                const float tick_1_sdf = max(interval_1d_sdf(uv.x, tick_start_x, tick_start_x + tick_length), ticks_1_deg.z);
                const float tick_10_sdf = max(interval_1d_sdf(uv.x, tick_start_x, tick_start_x + tick_length * 2), ticks_10_deg.z);
                const float window_sdf = interval_1d_centered_sdf(uv.y, 0., 11. / 180. * pi);
                const float ticks_sdf = max(min(tick_1_sdf, tick_10_sdf), window_sdf);

                // Display 2 digit degree count
                if(window_sdf < 0 && interval_1d_sdf(uv.x, legend_start_x, legend_start_x + 3 * glyph_scale * glyph_max_box_size.x) < 0) {
                    const float legend_elevation_rad = ticks_10_deg.x;
                    float2 legend_origin = float2(legend_start_x, legend_elevation_rad - elevation_at_0) + glyph_scale * glyph_max_box_size * float2(0, -0.35);

                    uint digit_10 = (uint) round(abs(legend_elevation_rad) * 18. / pi); // [-9, 9]

                    UNITY_FLATTEN if(legend_elevation_rad < 0) {
                        legend_origin = renderer.add_right(11 /* '-' */, uv, legend_origin, glyph_scale);
                    } else {
                        legend_origin.x += glyph_definition_table[11].advance * glyph_scale;
                    }

                    legend_origin = renderer.add_right(digit_10, uv, legend_origin, glyph_scale);
                    renderer.add_right(0, uv, legend_origin, glyph_scale);
                }          
                return ticks_sdf;
            }

            float azimuth_display_sdf(float2 uv, float azimuth_at_0, inout GlyphRendererOld renderer) {
                const float tick_start_y = 0.2;
                const float tick_length = 0.01;
                const float legend_start_y = tick_start_y + tick_length * 2.1;
                const float glyph_scale = 0.0003;
                
                // UVs are sin angle from normal ~ angle for center, in radiants
                const float pixel_azimuth = azimuth_at_0 + uv.x;

                // Tick marks : vertical column & distance to nearest tick using round(y)
                const float3 ticks_1_deg = closest_step_positioning(pixel_azimuth, 1.);
                const float3 ticks_10_deg = closest_step_positioning(pixel_azimuth, 10.);
                const float tick_1_sdf = max(interval_1d_sdf(uv.y, tick_start_y, tick_start_y + tick_length), ticks_1_deg.z);
                const float tick_10_sdf = max(interval_1d_sdf(uv.y, tick_start_y, tick_start_y + tick_length * 2), ticks_10_deg.z);
                const float window_sdf = interval_1d_centered_sdf(uv.x, 0., 15. / 180. * pi);
                const float ticks_sdf = max(min(tick_1_sdf, tick_10_sdf), window_sdf);

                // Display 2 digit degree count
                if(window_sdf < 0 && interval_1d_sdf(uv.y, legend_start_y, legend_start_y + glyph_scale * glyph_max_box_size.y) < 0) {
                    const float legend_azimuth_rad = ticks_10_deg.x;
                    float2 legend_origin = float2(legend_azimuth_rad - azimuth_at_0, legend_start_y) + glyph_scale * glyph_max_box_size * float2(-1.5, 0.1);

                    const float legend_azimuth_10deg = legend_azimuth_rad * 18. / pi;
                    uint n = (uint) glsl_mod(legend_azimuth_10deg, 36.); // [0, 35]
                    uint digit_10 = n % 10;
                    uint digit_100 = n / 10;

                    legend_origin = renderer.add_right(digit_100, uv, legend_origin, glyph_scale);
                    legend_origin = renderer.add_right(digit_10, uv, legend_origin, glyph_scale);
                    renderer.add_right(0, uv, legend_origin, glyph_scale);
                }          
                return ticks_sdf;
            }

            ////////////////////////////////////////////////////////////////////////////////////
            // Composition

            fixed4 fragment_stage (FragmentInput input) : SV_Target {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                // i.{aligned/rotating}_uv_{x/y}_os are vectors in a plane facing the view.
                // We want a measure of angle to view dir ~ sin angle to view dir = cos angle to these plane vectors.
                const float3 ray_ws = normalize(input.camera_to_geometry_ws);
                const float2 aligned_uv = float2(dot(ray_ws, input.hud_data.aligned_uv_x_ws), dot(ray_ws, input.hud_data.aligned_uv_y_ws));

                const float screenspace_scale_of_uv = compute_screenspace_scale_of_uv(aligned_uv); // Both uv sets should have the same scale

                Font font = Font::init();
                float ui_sd = sdf_crosshair_0thickness(aligned_uv);
                font.draw_glyphs_6x12(aligned_uv, input.hud_data.glyphs, _Measurement_Digit_block_Position, _Font_Size);

                GlyphRendererOld renderer_old = GlyphRendererOld::create();
                float sdf = 1000 * elevation_display_sdf(aligned_uv, input.hud_data.elevation_radiants, renderer_old);
                sdf = min(sdf, 1000 * azimuth_display_sdf(aligned_uv, input.hud_data.azimuth_radiants, renderer_old));
                sdf = min(sdf, 3 * renderer_old.sdf(0.15)); // Scale for sharpness

                const float sd = min(ui_sd - _UI_Thickness, font.sdf());
                float opacity = sdf_blend_with_aa(sd, screenspace_scale_of_uv);
                
                // TODO remove legacy blurring
                const float positive_distance = max(0, sdf);
                opacity = max(opacity, 1. - positive_distance * positive_distance);
                
                return _Color * opacity;
            }

            ENDCG
        }
    }
}

// Made by Lereldarion (https://github.com/lereldarion/unity-shaders)
// Free to redistribute under the MIT license

// Overlay at optical infinity (reflecting sight), with crosshair, rangefinder distance, and worldspace compass.
// Attached to a flat surface with uniform UVs (like a quad), it will orient itself with the tangent space.
Shader "Lereldarion/Overlay/HUD" {
    Properties {
        [Header(HUD)]
        [HDR] _Color("Emissive color", Color) = (0, 1, 0, 1)
        [HideInInspector] _MSDF_Glyph_Atlas ("MSDF glyph texture", 2D) = "" {}

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
        Blend OneMinusDstColor OneMinusSrcAlpha // Overlay transparency + Blacken when overlayed on white stuff
        ZWrite Off
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

            uniform fixed4 _Color;
            uniform float _Overlay_Fullscreen;
            uniform float _Overlay_Screenspace_Vertex_Reorder;

            uniform SamplerState sampler_clamp_bilinear;
            uniform Texture2D<float3> _MSDF_Glyph_Atlas;

            uniform float _VRChatMirrorMode;
            uniform float _VRChatCameraMode;

            static const float _UI_Thickness = 0.001;
            static const float _Crosshair_Circle_Radius = 0.03;
            static const float _Crosshair_Tick_Length = 0.04;
            static const float _Font_Size = 0.023;
            static const float2 _Measurement_Digit_block_Position = float2(-0.3, -0.2);
            static const float _Compass_Tick_Start = 0.2;
            static const float _Compass_Tick_Length = 0.01;

            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);

            // Utils
            float2 pow2(float2 v) { return v * v; }
            float round_to_scale(float v, float scale) { return scale * round(v / scale); }
            float glsl_mod(float x, float y) { return x - y * floor(x / y); }
            static const float pi = UNITY_PI;
            static const float one_deg_rad = pi / 180;
            static const float nan = asfloat(uint(-1)); // 0xFFF...FFF should be a quiet NaN
            static const float f32_infinity = asfloat(0x7f800000);

            // Replacement for missing invP by d4rkpl4y3r (https://gist.github.com/d4rkc0d3r/886be3b6c233349ea6f8b4a7fcdacab3)
            float4 ClipToViewPos(float4 clipPos) {
                float4 normalizedClipPos = float4(clipPos.xyz / clipPos.w, 1);
                normalizedClipPos.z = 1 - normalizedClipPos.z;
                normalizedClipPos.z = normalizedClipPos.z * 2 - 1;
                float4x4 invP = unity_CameraInvProjection;
                // do projection flip on this, found empirically
                invP._24 *= _ProjectionParams.x;
                // this is needed for mirrors to work properly, found empirically
                invP._42 *= -1;
                float4 viewPos = mul(invP, normalizedClipPos);
                // and the y coord needs to flip for flipped projection, found empirically
                viewPos.y *= _ProjectionParams.x;
                return viewPos;
            }

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
            float sdf_disk(float2 p, float radius) { return length(p) - radius; }
            float sdf_circle(float2 p, float radius) { return abs(sdf_disk(p, radius)); }
            float sdf_segment_x(float2 p, float from, float to) {
                float2 closest_segment_point = float2(clamp(p.x, from, to), 0);
                return distance(p, closest_segment_point);
            }
            float sdf_interval_1d_centered(float x, float center, float radius) { return abs(x - center) - radius; }

            ////////////////////////////////////////////////////////////////////////////////////
            // MSDF monospace glyph rendering.
            // https://github.com/Chlumsky/msdfgen
            // Charset : ['0', '9'] "XYZ+-.∞ø:◁⇥f" GeistMono-Regular uniform grid rectangle texture
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
                void draw_glyph(float2 p, uint glyph, float2 offset_bbox) {
                    const float scale = ascender_px / _Font_Size;
                    const float2 px = p * scale - offset_bbox * glyph_bbox_px;
                    if(all(0 <= px && px <= glyph_bbox_px)) {
                        sampling_offset_px = px;
                        sampling_inverse_scale = 1 / scale;
                        sampling_atlas_id = glyph;
                    }
                }
                void draw_3_glyphs(float2 p, uint3 glyphs, float2 offset_bbox) {
                    const float scale = ascender_px / _Font_Size;
                    const float2 px = p * scale - offset_bbox * glyph_bbox_px;
                    if(all(0 <= px && px <= glyph_bbox_px * float2(3, 1))) {
                        float column = floor(px.x / glyph_bbox_px.x);
                        sampling_offset_px = px - float2(column * glyph_bbox_px.x, 0);
                        sampling_inverse_scale = 1 / scale;
                        sampling_atlas_id = glyphs[column];
                    }
                }
                void draw_glyphs_6x12(float2 p, uint4 glyphs[3], float2 offset_bbox) {
                    // glyphs : X/Z=00fedcba, Y/W=00lkjihg
                    const float scale = ascender_px / _Font_Size;
                    const float2 px = p * scale - offset_bbox * glyph_bbox_px;
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
                nointerpolation float3 polar_x_ws : POLAR_X;
                nointerpolation float3 polar_y_ws : POLAR_Y;

                // "X -123456.89" : 12 glyphs, 5 bits each. 6 per u32. 
                nointerpolation uint4 glyphs[3] : GLYPHS; // X/Z=00fedcba, Y/W=00lkjihg, [i].XY = world pos, [i].ZW = {far plane, range, fps}

                nointerpolation float azimuth_radiants : AZIMUTH;
                nointerpolation float elevation_radiants : ELEVATION;

                static HudData compute(float3 forward_normal_ws) {
                    HudData hd;

                    // Axis of polar-like coordinate, aligned to world up
                    const float3 up_direction_ws = float3(0, 1, 0);
                    const float3 horizontal_tangent = normalize(cross(forward_normal_ws, up_direction_ws));
                    hd.polar_x_ws = horizontal_tangent * -1 /* needed but not sure why ; ht should go to the right */;
                    hd.polar_y_ws = cross(horizontal_tangent, forward_normal_ws);

                    // World azimuth and elevation of the surface forward normal
                    const float3 east_ws = float3(1, 0, 0);
                    const float3 north_ws = float3(0, 0, 1);
                    const float angular_dist_to_north_0_pi = acos(dot(horizontal_tangent, east_ws));
                    hd.azimuth_radiants = pi - angular_dist_to_north_0_pi * sign(dot(horizontal_tangent, north_ws)); // 0 at north, pi/2 east, pi south, 3pi/2 west
                    const float angular_dist_to_up_0_pi = acos(dot(forward_normal_ws, up_direction_ws));
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
                    const float range_ws = LinearEyeDepth(depth_texture_value) / sample_point_cs.w; // Cannot detect presence of Depth texture, this may be garbage.

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
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct FragmentInput {
                float4 position : SV_POSITION;
                float3 ray_ws : RAY_WS;
                HudData hud_data;

                UNITY_VERTEX_OUTPUT_STEREO
            };

            void vertex_stage (VertexInput input, uint vertex_id : SV_VertexID, out FragmentInput output) {
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                float3 forward_normal_ws;
                bool compute_hud_data = true;

                if(_Overlay_Fullscreen == 1 && _VRChatMirrorMode == 0 && _VRChatCameraMode == 0) {
                    // Fullscreen mode : cover the screen with a quad by redirecting existing vertices
                    if(vertex_id < 4) {
                        float2 ndc = vertex_id & uint2(2, 1) ? 1 : -1; // [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                        if(_Overlay_Screenspace_Vertex_Reorder && (vertex_id & 1)) { ndc.x *= -1; }
                        output.position = float4(ndc, UNITY_NEAR_CLIP_VALUE, 1);

                        const float3 forward_vs = float3(0, 0, -1);
                        const float4 position_vs = ClipToViewPos(output.position);
                        output.ray_ws = mul(unity_MatrixInvV, position_vs.xyz / position_vs.w);
                        forward_normal_ws = normalize(mul(unity_MatrixInvV, forward_vs));
                    } else {
                        output.position = nan.xxxx; // Vertex discard
                        output.ray_ws = 0;
                        forward_normal_ws = 0;
                        compute_hud_data = false;
                    }
                } else {
                    const float3 position_ws = mul(unity_ObjectToWorld, float4(input.position_os.xyz, 1)).xyz;
                    output.position = UnityWorldToClipPos(position_ws);
                    output.ray_ws = position_ws - _WorldSpaceCameraPos;
                    const float3 normal_ws = UnityObjectToWorldNormal(input.normal_os);
                    forward_normal_ws = dot(normal_ws, output.ray_ws) >= 0 ? normal_ws : -normal_ws;
                }

                if(compute_hud_data) {
                    output.hud_data = HudData::compute(forward_normal_ws);
                } else {
                    output.hud_data = (HudData) 0;
                }
            }

            ////////////////////////////////////////////////////////////////////////////////////
            // Sight

            float sdf_crosshair(float2 p) {
                const float circle = abs(sdf_disk(p, _Crosshair_Circle_Radius));

                // 4 symmetric segments. Use symmetry to only look at east one.
                p = abs(p);
                p = p.x > p.y ? p : p.yx;
                const float cross = sdf_segment_x(p, _Crosshair_Circle_Radius - 0.5 * _Crosshair_Tick_Length, _Crosshair_Circle_Radius + 0.5 * _Crosshair_Tick_Length);
                return min(circle, cross);
            }

            ////////////////////////////////////////////////////////////////////////////////////
            // Compass

            float sdf_elevation(float2 p, float elevation_at_0, inout Font font, float polar_screen_scale) {
                float sd = f32_infinity;

                // Restrict to bounds, but allow gap for AA
                const float vertical_bounds = sdf_interval_1d_centered(p.y, 0, 11 * one_deg_rad);
                if(vertical_bounds < polar_screen_scale) {
                    // polar angle from normal, in radiants
                    p.y += elevation_at_0;
    
                    // Positionning
                    const float closest_1deg = one_deg_rad * round(p.y / one_deg_rad);
                    const float closest_10deg_unit = round(p.y / (10 * one_deg_rad)); // 1 = 10°
                    const float closest_10deg = (10 * one_deg_rad) * closest_10deg_unit;
                    const bool is_10deg_tick = abs(closest_10deg - closest_1deg) < 0.5 * one_deg_rad;
    
                    // Ticks
                    const float tick_length = _Compass_Tick_Length * (is_10deg_tick ? 2 : 1);
                    const float ticks = sdf_segment_x(p - float2(_Compass_Tick_Start, closest_1deg), 0, tick_length);
                    sd = max(ticks, vertical_bounds);

                    // Legend
                    uint digit_10 = uint(abs(closest_10deg_unit));
                    if(digit_10 > 9) { digit_10 = 18 - digit_10; } // 10 -> 8, after poles
                    const uint3 glyphs = uint3(closest_10deg_unit < 0 ? Font::minus : Font::space, Font::zero + digit_10, Font::zero);
                    font.draw_3_glyphs(p - float2(_Compass_Tick_Start + 2.5 * _Compass_Tick_Length, closest_10deg), glyphs, float2(0, -0.5));
                }
                return sd;
            }

            float sdf_azimuth(float2 p, float azimuth_at_0, inout Font font, float polar_screen_scale) {
                float sd = f32_infinity;

                const float horizontal_bounds = sdf_interval_1d_centered(p.x, 0, 20 * one_deg_rad);
                if( horizontal_bounds < polar_screen_scale) {
                    // polar angle from normal, in radiants
                    p.x += azimuth_at_0;
                    
                    // Positionning
                    const float closest_1deg = one_deg_rad * round(p.x / one_deg_rad);
                    const float closest_10deg_unit = round(p.x / (10 * one_deg_rad)); // 1 = 10°
                    const float closest_10deg = (10 * one_deg_rad) * closest_10deg_unit;
                    const bool is_10deg_tick = abs(closest_10deg - closest_1deg) < 0.5 * one_deg_rad;
    
                    // Ticks
                    const float tick_length = _Compass_Tick_Length * (is_10deg_tick ? 2 : 1);
                    const float ticks = sdf_segment_x(p.yx - float2(_Compass_Tick_Start, closest_1deg), 0, tick_length);
                    sd = max(ticks, horizontal_bounds);

                    // Legend
                    const uint n = (uint) glsl_mod(closest_10deg_unit, 36); // [0, 35]
                    const uint3 glyphs = uint3(n / 10, n % 10, 0) + Font::zero;
                    font.draw_3_glyphs(p - float2(closest_10deg, _Compass_Tick_Start + 2.5 * _Compass_Tick_Length), glyphs, float2(-1.5, 0));
                }
                return sd;
            }

            ////////////////////////////////////////////////////////////////////////////////////

            fixed4 fragment_stage (FragmentInput input) : SV_Target {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                
                // We want polar (angle) coordinates with x/y aligned with normal and world up.
                // We want a measure of angle to view dir, with sin angle to view dir = cos angle to these plane vectors.
                // This is not linear, but linear enough around the normal, and with a pleasing round deformation.
                const float3 ray_ws = normalize(input.ray_ws);
                const float2 polar = asin(float2(dot(ray_ws, input.hud_data.polar_x_ws), dot(ray_ws, input.hud_data.polar_y_ws)));

                const float polar_screen_scale = compute_screenspace_scale_of_uv(polar);

                // Generate UI elements (ticks, 0 thickness), and register glyphs.
                Font font = Font::init();
                float ui_sd = sdf_crosshair(polar);
                font.draw_glyphs_6x12(polar - _Measurement_Digit_block_Position, input.hud_data.glyphs, 0);
                ui_sd = min(ui_sd, sdf_elevation(polar, input.hud_data.elevation_radiants, font, polar_screen_scale));
                ui_sd = min(ui_sd, sdf_azimuth(polar, input.hud_data.azimuth_radiants, font, polar_screen_scale));

                // Combine both sdfs
                const float sd = min(ui_sd - _UI_Thickness, font.sdf());                
                return _Color * sdf_blend_with_aa(sd, polar_screen_scale);
            }

            ENDCG
        }
    }
}

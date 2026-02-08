# Changelog

## [unreleased]
- Grid : remove broken grid for infnite distance (skybox)

## 1.4.0 - 08/02/2026
- Add a changelog with previous release history
- Remove materials from released assets. Put them in a `Samples/` directory for backward compatibility
- Improved Hidden shader with NaN clip space positions and the Disabled Always pass trick
- Depth reconstruction : use d4rkpl4y3r technique of patching unity_CameraInvProjection (https://gist.github.com/d4rkc0d3r/886be3b6c233349ea6f8b4a7fcdacab3). Tested to be working in Desktop & VR (PC DX11)
- Remove geometry pass from many overlays (fullscreen mode)
- Gamma Adjust : toggle for transmitting emission or not
- HUD : rewritten to use proper SDF UI with MSDF text. Removed geometry pass. Added zoom option.

## 1.3.4 - 28/12/2025
- Fix missing mip streaming on hud sdf texture

## 1.3.3 - 18/12/2025
- Fix vertexcount issues with lighting debug

## 1.3.2 - 02/11/2025
- Improve package metadata. No functional change.

## 1.3.1 - 02/11/2025
Lighting debug
- Point pixel light : use icosahedron shape.
- Fix gizmo size for short range lights

## 1.3.0 - 29/10/2025
- Lighting debug : Increased gizmos complexity to have more recognizable shapes.
- Line gizmos now can exceed the geometry vertex count some may not be rendered in case of high light volume light counts.
  However this should only happen at very high light counts (>80), which would be unreadable anyway...

## 1.2.0 - 27/10/2025
- Added support for light volumes v2.0 lights.
- Harmonized gizmos between unity, LTCGI and LV. They will use the same style per light type, with visual complexity decreasing, and LV using dashes.

## 1.1.1 - 28/05/2025
- Fix debug lighting

## 1.1.0 - 28/05/2025
- Added lighting debug shader

## 1.0.8 - 11/05/2025
- Added TBN debug shader

## 1.0.7 - 18/02/2025
- Fix Bloom explosions on positive gamma adjust with emission.

## 1.0.6 - 01/11/2024
- improved material property presentation with headers
- improved grid rendering : now alpha-blended, and using crisp lines thanks to bgolus article

## 1.0.5 - 22/10/2024
Fix orientation problems with HUD shader :
- removed "uv swap" toggle that is not needed anymore, now that the tangent space is used correctly.
- fix fullscreen mirroring of HUD interface if the mesh is behind the camera ; hud data is computed from the fullscreen position instead of the mesh one. 

## 1.0.4 - 13/10/2024
- Consolidate documentation into the main github readme

## 1.0.3 - 13/10/2024
- Improved package metadata

## 1.0.2 - 13/10/2024
- Simplify directory layout.
- Add Material for Hidden shader.

## 1.0.1 - 13/10/2024
Bugfix: added minimum supported Unity version to remove VCC warning

## 1.0.0 - 13/10/2024
Added shaders :
- Gamma Adjust
- Depth reconstruction : Normals, Wireframe, Grid
- HUD

# Changelog

## [Unreleased]
- Remove materials from released assets. Put them in a `Samples/` directory for backward compatibility
- Improved Hidden shader with NaN clip space positions and the Disabled Always pass trick
- Depth reconstruction : use d4rkpl4y3r technique of patching unity_CameraInvProjection (https://gist.github.com/d4rkc0d3r/886be3b6c233349ea6f8b4a7fcdacab3). Tested to be working in Desktop & VR (PC DX11)
- Remove geometry pass from many overlays (fullscreen mode)
- Gamma Adjust : toggle for transmitting emission or not
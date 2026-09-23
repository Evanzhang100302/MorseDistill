# Per-dataset config for the two datasets shipped in this release.
# EMB is an unused placeholder: every run in this repo uses --no_vlm, which
# zeroes the VLM projector output, so the embedding lookup table is never
# actually read. It is kept only because --emb_file is a required argument.
ds_config() {
  case $1 in
    Wildfire_CA)
      VZ=8; EMB="embeddings/tile_embeddings_Wildfire_CA_z8.pt"; SZ=8 ;;
    # NASA FIRMS Canada 2023 hotspot data: acquisition times are real
    # satellite overpass times, unlike the linspace-generated sets we do not
    # ship here.
    Canada_fire)
      VZ=7; EMB="embeddings/tile_embeddings_Canada_fire_z7.pt"; SZ=9 ;;
    *) echo "unknown dataset $1 -- this release only ships Wildfire_CA and Canada_fire" >&2; return 1 ;;
  esac
  STF="st_morse_features/st_morse_features_$1_z${SZ}_b16.pt"
}

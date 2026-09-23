# Per-dataset config for the datasets shipped in this release.
# EMB is an unused placeholder: every run in this repo uses --no_vlm, which
# zeroes the VLM projector output, so the embedding lookup table is never
# actually read. It is kept only because --emb_file is a required argument.
ds_config() {
  case $1 in
    Wildfire_CA)
      # Kaggle US wildfire panel (per-state cut).
      VZ=8; EMB="embeddings/tile_embeddings_Wildfire_CA_z8.pt"; SZ=8 ;;
    Canada_fire)
      # Kaggle mirror of NASA FIRMS 2023 Canada hotspot detections
      # (https://www.kaggle.com/datasets/brsdincer/canada-wildfire-2023-hotspot-data).
      # Acquisition times are real satellite overpass times, unlike the
      # linspace-generated sets we do not ship here.
      VZ=7; EMB="embeddings/tile_embeddings_Canada_fire_z7.pt"; SZ=9 ;;
    eMAS_fire)
      # NASA eMAS (enhanced MODIS Airborne Simulator) fire detections.
      # NOTE: the shipped train/val/test split is chronological and the
      # held-out region is only partially covered by training (see paper
      # appendix). The teacher is unstable across seeds on this split --
      # seed 2026 (shipped in checkpoints/) trains cleanly; seeds 42 and 888
      # can collapse. This is a property of the released split, not a bug.
      VZ=7; EMB="embeddings/tile_embeddings_eMAS_fire_z7.pt"; SZ=12 ;;
    *) echo "unknown dataset $1 -- this release ships Wildfire_CA, Canada_fire, and eMAS_fire" >&2; return 1 ;;
  esac
  STF="st_morse_features/st_morse_features_$1_z${SZ}_b16.pt"
}

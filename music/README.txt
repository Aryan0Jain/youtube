Place royalty-free music bed MP3s here. Required filenames:

  dark_ambient.mp3           ← horror
  cinematic_orchestral.mp3   ← what_if
  high_energy_electronic.mp3 ← shock_facts
  neutral_corporate.mp3      ← comparison
  playful_upbeat.mp3         ← quiz
  epic_buildup.mp3           ← ranking
  investigative_jazz.mp3     ← myth_busting

Sources (free, no attribution required for YouTube):
  - Pixabay Music: https://pixabay.com/music/
  - YouTube Audio Library: https://studio.youtube.com/channel/UC.../music

On GCE VM, sync from GCS after initial upload:
  gsutil -m cp "gs://$GCS_BUCKET_NAME/music/*" ./music/

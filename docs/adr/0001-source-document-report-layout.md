# Source Document Report Layout

Status: accepted

Implementation status: deferred beyond the September 2026 consolidation baseline.

The target layout puts reviewable outputs under one directory per source document,
including single-document runs. Stable document-owned paths make provenance assets
unambiguous and give replay and evaluation a stable document identity to target.

The current implementation isolates OCR, images and caches for multiple PDFs under
`documents/<input-stem>/`, but keeps aggregate requirement exports at the run root.
Single-document runs keep their artifacts directly at the run root. This layout is
preserved for the consolidation baseline; this ADR does not describe behavior
already implemented. Moving the review exports requires a separate change to the
writers, manifests, replay/evaluation consumers and their compatibility tests.

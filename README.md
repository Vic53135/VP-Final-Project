# VP-Final-Project

This repository contains the final project for a **Video Processing** course.  
The project focuses on **learned image compression** and investigates improved entropy modeling strategies based on the DCAE framework.


## Method Description

- **DCAE (Baseline)**  
  The original Deep Contextual Autoencoder for learned image compression, used as the baseline for comparison.

- **MethodA (Channel Context)**  
  Extends the entropy model by improving channel context modeling to better capture global spatial dependencies in latent representations.

- **MethodB (Spatial + Channel Context)**  
  Builds upon MethodA by explicitly modeling inter-slice channel dependencies using RWKV-based linear attention, enabling efficient long-range dependency modeling with linear complexity.

## Notes

- All methods are trained and evaluated under consistent experimental settings for fair comparison.
- The project emphasizes architectural understanding and experimental analysis rather than large-scale training.
- This repository is intended for **academic and educational purposes** as a course final project.


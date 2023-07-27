# Secure-Robust-FL
Privacy preserving robust FL with clustering 
Currently, clusters are not employed (Single parameters server).

To run private training run main.py with "--private_client_training --sigma=<YOUR_SIGMA_VALUE> --clip_val=<CLIP_VAL>"
Currently, sigma value and its corresponding DP guarantees are calculated offline. This will be automated later.
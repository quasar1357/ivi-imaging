## Parameters/adjustments for workflow

- Pre-processing (maybe for cell masks instead, excluding background)
    - Denoising --> no
    - Other filters... --> no
    - Deconvolution ? --> Ben: don't do it...
    - Convpaint ?
- Normalization across samples -> need controls! (otherwise voluntary...)
    - Possible to acquire background signal within sample (e.g. remove cells from one half) ???

- (Segmentation --> only if not working properly)

- Dilation/erosion for mean --> try...
- Alternatives to mean signal (median, functions, filters...) --> mean = usual

- Threshold for binning --> use mocks (avg_ctrl + 3 * sd_ctrl)
- (>2 bins; evtl. with +/- around threshold)





Mit Ben:
für signal: normal = mean; aber evtl. auch mal upper_quartile o.ä. probieren
mid_slice statt projection ? --> aber vermutlich auch nicht repräsentativer
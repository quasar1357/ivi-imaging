## Python scripts

- file handling
	- regex: `re.findall('.+_.+_.+_(.+)_.+\.tif',filename)`
- saving intermediate results, such as segmentation: `cellpose.io.masks_flows_to_seg(imgs, masks, flows, diams, to_segment, channels)`
- saving summary images such as signals with segmentation
- 3D segmentation (not in script, but Cellpose website)
- Cell properties, but not really used:
	- `regions = skimage.measure.regionprops_table(skimage.measure.label(maski), intensity_image=measi, properties=('label','area',"perimeter",'mean_intensity', "feret_diameter_max", "major_axis_length", "minor_axis_length"))`
	- `regions = pd.DataFrame(regions)`
	- --> area | perimeter | mean_intensity | feret_diameter_max | major_axis_length | minor_axis_length
---
- `inf_lvl = np.log10(mean)` --> `mock_mean = mean(inf_lvl[mocks])` --> `inf = inf_lvl > mock_mean + 3 mock_sd`
	- --> `results_count["total_cells"]= 1.0 --> groupby(["experiment", "virus", "replicate", "channel"]).sum().reset_index()`
	- --> gives sum of infected (each is 0 or 1) and total cells (each is 1) for each replicate
- ( `results_infection.groupby(["experiment", "virus", "replicate", "channel"]).mean().reset_index()` --> mean of each value across cells in each replicate )
---
- `add_stat_annotation(ax, data=all_results_summary, x=x, y=y, hue=hue, box_pairs=box_pairs, test='t-test_paired',  comparisons_correction= None, loc='inside', verbose=2)` --> significance of differences added to sns plot


**=> ' log10 of signal --> +3*SD from mock_mean ' = infected (instead of finding threshold from distribution)**



## R script

### General ideas

- Clean data from too small cells

- Thresholding for infection:
	`mutate(Cy5_mean_intensity=Cy5_mean_intensity, GFP_mean_intensity=GFP_mean_intensity,
                  infected_cell = ifelse(GFP_mean_intensity > mean_sd, 1, 0))`
- Thresholding for cilia (comparison with mock makes no sense):
	- `mutate(loc = (locmodes(cilia, mod0 = 2, display = FALSE))$locations[2])`
		- For each group (i.e., per image), it calculates modes in the cilia intensity distribution using the `locmodes()` function from the multimode package.
		- `mod0 = 2` tells it to look for two modes (bimodal distribution).
		- It extracts the second mode (`$locations[2]`) and stores it as `loc`
	- `mutate(ciliated_cell = ifelse(cilia > loc , 1, 0))`

### Cilia (cilia/cell between species in mock samples)

1. **Get fractions**
- Isolating mock-infected samples
- Counting total and ciliated cells
- Computing the fraction of ciliated cells per donor
- Summarizing these ratios per species with mean, SEM, and SD — perfect for downstream plotting (e.g. bar plots with error bars)

2. **Statistical tests**
- Shapiro-Wilk test to test for normality of the distribution
	- Shapiro-Wilk assumes you're testing one group — if you're comparing ratios by species, you may want to test within each species
- Perform a pairwise paired t-test comparing ratio across levels of species
	- For a paired t-test, the pairing variable (like donor) must be consistent across species. If donors are not shared between species, `paired = TRUE` is not appropriate

### Cell counts

1. **Count**
- Filter infected samples (i.e., non-mock)
- Count cells per image and then per donor --> average per donor
- Calculate mean, SD, and SEM of cell counts per condition (group of species/temp/virus/infection status)

2. **Normalize by area**
- Manually create a table of species-specific image dimensions
- Compute image area
- Normalize cell counts to cells/mm² ( mutate(cell_mm2 = cell_count_donor / mm2) )
- Summarize across experimental groups with mean cell density, SD and SEM ...

3. **Statistical tests (on normalized data)**
- Apply a Shapiro-Wilk test on cell_mm2 of uninfected cells to test for normality within each species
- Pairwise t-test 1: Infected vs Uninfected
If your dataset includes different donors or samples between infected and uninfected conditions, this assumption breaks, and you should use:
`pairwise_t_test(cell_mm2 ~ infected_cell, paired = FALSE)`
- Pairwise t-test 2: Compares 33°C vs 37°C within infected samples

4. **Final summary (on NON-normalized data !)**
- Image-level summary of infected ciliated/non-ciliated cells
	- Filter only infected cells (`infected_cell == "1"`)
	- Group by all relevant metadata + ciliated status + image
	- Count cells per image
- Donor-level summary
	- Average the per-image cell counts across all images from the same donor
	- Ensure that for every combination of species, temp, virus, and ciliation, you have a count — even if it's 0
- Condition-level summary
	- Average across donors for each condition and ciliated status
	- Ensure no condition is left out, even if there were no cells counted for a certain group
- **==> For a given species, temperature, virus, and ciliated status (1 or 0), what's the average number of infected cells per donor? (NOT normalized by area...)**

### Details t-tests for cell counts:
```
Test 1:
    For each combination of species, virus, and temperature, you're comparing:
        Cell density of infected cells (infected_cell == 1)
        VS non-infected cells (infected_cell == 0)
    Within the same condition group, i.e., same donor, virus, temp, and species.
🧠 Interpretation:
    "Among all samples that were exposed to virus (i.e., infected), is there a difference in cell density between cells that were actually infected and those that remained uninfected, under otherwise identical conditions?"
	--> This is not comparing mock vs infected samples. It's comparing infected vs bystander (non-infected) cells within infected samples.
```
```
Test 2:
    For each combination of species and virus, you're comparing:
        Cell density of infected cells at 33°C
        VS infected cells at 37°C
    Only considering infected_cell == 1 (so non-infected cells are excluded).
🧠 Interpretation:
    "Does the temperature affect the density of infected cells (per mm²), within the same virus and species?"
	--> This answers: "Are there more (or fewer) infected cells at 33°C compared to 37°C, assuming the same donor/species/virus pairing?"
```
Test | Comparing | Filtered On | Grouped By | Purpose
------|-----------|-------------|------------|--------
pwc1 | Infected vs Non-infected cell densities | All infected samples (C, D) | species, virus, temperature | Do infected cells accumulate differently than bystanders?
pwc2 | Infected cell densities at different temperatures | infected_cell == 1 only | species, virus | Does temperature affect infection outcome (cell density)?

### Issues:

- Species_IS <- `subset(Species, infection %in% c("C", "D"))` NOT `infection == c("C", "D")`
- Normalizations... (only partially done/used)
- Paired t-test appropriate?

To double-check, run this:

`table(normalization_cells$species, normalization_cells$donor, normalization_cells$infected_cell)`

--> Look for whether each donor appears with both infection statuses
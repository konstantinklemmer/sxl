#
## ACCESS PETREL DATA ##
#


# SETUP #


#CHECK, INSTALL AND LOAD REQUIRED PACKAGES
pkgs <- c("sp","rgdal","spdep","sf","rlist","elevatr","rlang","raster","spm","reporttools")
for (pkg in pkgs) {
  if(pkg %in% rownames(installed.packages()) == FALSE) {install.packages(pkg)
    lapply(pkgs, require, character.only = TRUE)}
  else {
    lapply(pkgs, require, character.only = TRUE)}
}
rm(pkg,pkgs)
#SET WORKING DIRECTORY TO FILEPATH OF SCRIPT (PREFERRED DIRECTORY WHEN CLONING THE REPOSITORY)
#THIS WILL ONLY WORK WHEN USING R STUDIO, ELSE SET WD MANUALLY
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

#HELPER FUNCTIONS
matrify = function(raster_list) { #CONVERT RASTER LIST TO MATRIX LIST
  for (i in seq(1,length(raster_list))) {
    raster_list[[i]] = as.matrix(raster_list[[i]])
  }
  return(raster_list)
}

# DATA #

#Access petrel grid data (no download is required for this dataset)
data(petrel.grid)
#Delete incomplete rows / columns
petrel.grid = petrel.grid[petrel.grid$long < 130.02223003,]
petrel.grid = petrel.grid[petrel.grid$lat > -11.67239117,]
#Define size of grid cells
n = 32
#Define ncol and nrow for raster
n_col = floor(length(unique(petrel.grid$long)) / n) * n
n_row = floor(length(unique(petrel.grid$lat)) / n) * n
#Set coordinates
coordinates(petrel.grid) = ~long+lat
#Create empty raster
petrel = raster()
extent(petrel) = extent(petrel.grid)
ncol(petrel) = n_col # this is one way of assigning cell size / resolution
nrow(petrel) = n_row

# PROCESS DATA

#Combine data (underwater relief) and empty raster
petrel.rel = rasterize(petrel.grid, petrel, petrel.grid$relief, fun = mean)
#Crop raster into 32 * 32 pixel tiles
a = aggregate(raster(petrel.rel), n)
p = as(a, 'SpatialPolygons')
petrel_rel_grid_raster = lapply(seq_along(p), function(i) crop(petrel.rel, p[i]))
#Convert to matrix list
petrel_rel_grid_matrix = matrify(petrel_rel_grid_raster)
#Save matrix list as .JSON
#list.save(petrel_rel_grid_matrix, 'list_petrel.json')

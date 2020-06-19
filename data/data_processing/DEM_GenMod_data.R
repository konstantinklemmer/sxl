#
## ACCESS DEM DATA ##
#


# SETUP #


#CHECK, INSTALL AND LOAD REQUIRED PACKAGES
pkgs <- c("sp","rgdal","spdep","sf","rlist","elevatr","rlang","raster")
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

#Access "lake" shape from "elevatr" package - This does NOT require downloading the dataset directly
data(lake)
#Download DEM at zoom 10
dem = get_elev_raster(lake, z = 10)
#Preprocess (change CRS, trim and crop to square)
dem = projectRaster(dem, crs = "+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0")
dem = trim(dem)
dem = dem[2:1089, 2:1089,drop=FALSE] 
#Define size of grid cells
n = 32

# PROCESS DATA # 

#Crop raster into 32 * 32 pixel tiles
a = aggregate(raster(dem), n)
p = as(a, 'SpatialPolygons')
dem_grid_raster = lapply(seq_along(p), function(i) crop(dem, p[i]))
dem_grid_matrix = matrify(dem_grid_raster)
#Save matrix grid to .JSON
#list.save(dem_grid_matrix, 'list_dem.json')
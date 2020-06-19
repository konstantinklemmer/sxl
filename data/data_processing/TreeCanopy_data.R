#
## ACCESS TREE COVER DATA ##
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

half_size = function(matrix_list) { #REDUCE MATRIX SIZE BY HALF BY DELETING EVERY SECOND ROW / COLUMN
  for (i in seq(1,length(matrix_list))) {
    matrix_list[[i]] = matrix_list[[i]][seq(1,nrow(matrix_list[[i]]),2),seq(1,ncol(matrix_list[[i]]),2)]
  }
  return(matrix_list)
}

# DATA #

#Access tree cover data
#Data Source: http://earthenginepartners.appspot.com/science-2013-global-forest/download_v1.6.html
# The precise download link is: https://storage.googleapis.com/earthenginepartners-hansen/GFC-2018-v1.6/Hansen_GFC-2018-v1.6_lossyear_60N_110W.tif

GDALinfo("Hansen_GFC-2018-v1.6_lossyear_60N_110W.tif")
tree = raster("Hansen_GFC-2018-v1.6_lossyear_60N_110W.tif")
#Define size of grid cells
n = 64

# PROCESSING DATA #

#Crop raster into 64 * 64 pixel tiles
a = aggregate(raster(tree), n)
p = as(a, 'SpatialPolygons')
tree_grid_raster = lapply(seq_along(p), function(i) crop(tree, p[i]))
#Convert raster list to matrix list
tree_grid_matrix = matrify(tree_grid_raster)
#Save matrix list as .JSON
list.save(tree_grid_matrix, 'list_tree.json')


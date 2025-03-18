# instanseg_cells_bioptimus
File to create a docker, for detect cells and extract Optimus features associated it 


# Run the docker
````
 docker run --gpus all -it -v /mnt/d/test_biopsy/:/app/images genehetx/instanseg_cells_extractor_bioptimus:latest
 ````
 replace /mnt/t/test_biopsy by the patjh of your data


 ### Run instanseg

 You have an exemple how run the whole process in /app/script/run_all.py





 Warning: on the github repository the optimus model isn't available ( but it is inside the docker), but you could find at https://huggingface.co/bioptimus/H-optimus-0

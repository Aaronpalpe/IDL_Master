##########################################################
CARPETAS Y SCRIPTS
############# Versión 2.0. Enero 2026
#########################

###########################################################################################################################
slurm_scripts
#############
# Scrits de ejemplo para trabajos no interactivos con SBATCH
# Resnet50: 
- 3 iteraciones de entrenamiento. 
- Tamaño de batch limitado a 1 (Caso CPU) y a 32 (Casos GPU y multi-GPU). 
- Número de muestras limitadas a 8 (Caso CPU) y a 4096 (caso GPU y multi-GPU)
--
#############
COLA01 (CPU)
#############
Script principal: ./submit_resnet50_cpu.sh
// Envío a cola01 un ejemplo limitado de entrenamiento del modelo Resnet50
// Este script ejecuta este otro: run_resnet50_cpu.sbatch
// que tras configuración de lanzamiento apptainer, se ejecuta el programa PyTorch de entrenamiento: $HOME/accelerate_scripts/resnet50-train_cpu.py

Ejemplo de lanzamiento:
>> ./submit_resnet50_cpu.sh


#############
COLA02 (GPU)
#############
Script principal: ./submit_resnet50_gpu.sh
// Envío a cola02 un ejemplo limitado de entrenamiento del modelo Resnet50
// Este script ejecuta este otro: run_resnet50_mgpu.sbatch
// que tras configuración de lanzamiento apptainer, se ejecuta el programa PyTorch de entrenamiento: $HOME/accelerate_scripts/resnet50-train_mgpu.py

Ejemplo de lanzamiento:
>> NUM_GPUs=1 ./submit_resnet50_gpu.sh


Script principal: ./submit_resnet50_gpu_stats.sh
// Igual que el anterior (submit_resnet50_gpu.sh) pero ofrece detalle de estadísticas de profiling de la ejecución
Ejemplo de lanzamiento:
>> NUM_GPUs=1 ./submit_resnet50_gpu_stats.sh



#############
COLA03 (Multi-GPU)
#############
Script principal: ./submit_resnet50_gpu.sh
// Envío a cola03 un ejemplo limitado de entrenamiento del modelo Resnet50
// Este script ejecuta este otro: run_resnet50_mgpu.sbatch
// que tras configuración de lanzamiento apptainer, se ejecuta el programa PyTorch de entrenamiento: $HOME/accelerate_scripts/resnet50-train_mgpu.py

Ejemplos de lanzamiento (2 GPUs, 3 GPUs y 4 GPUs)
>> NUM_GPUs=2 ./submit_resnet50_gpu.sh
>> NUM_GPUs=3 ./submit_resnet50_gpu.sh
>> NUM_GPUs=4 ./submit_resnet50_gpu.sh


Script principal: ./submit_resnet50_gpu_stats.sh
// Igual que el anterior (submit_resnet50_gpu.sh) pero ofrece detalle de estadísticas de profiling de la ejecución a nivel de GPU (Rank)
Ejemplos de lanzamiento (2 GPUs, 3 GPUs y 4 GPUs)
>> NUM_GPUs=2 ./submit_resnet50_gpu_stats.sh
>> NUM_GPUs=3 ./submit_resnet50_gpu_stats.sh
>> NUM_GPUs=4 ./submit_resnet50_gpu_stats.sh





###########################################################################################################################
accelerate_scripts
#############
# Ficheros de ejemplo para ejecutar experimentos con Accelerate:

####
- Dockerfile: fichero para construir la imagen Docker con todas los paquetes y librerías necesarias para ejecutar Accelerate. Este fichero es el que se utilizó de base para crear la imagen para Apptainer del clúster ATLASv2. Se puede usar para que cada estudiante trabaje en local (su PC).

####
- Ficheros *.yaml: ejemplos de fichero de configuración de Accelerate para cpu, GPU e IPEX

####
- Ficheros *.py: ficheros con código PyTorch + Accelerate para inferencia (*-inf_*) y entrenamiento (*-train_*) de modelos (resnet50, bert, nlp) básicos de ejemplo con configuraciones limitadas (e.g., un número de épocas de entrenamiento pequeño).


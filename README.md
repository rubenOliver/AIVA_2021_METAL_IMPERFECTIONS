# AIVA_2021_METAL_IMPERFECTIONS

## Introducción 

El proyecto que se va a realizar consiste en detección de imperfecciones en metales, pudiendo ser estas imperfecciones de tipo moho, arañazos, y manchas. Además de la detección del tipo de imperfección que esté presente en la pieza de metal, en el caso de los arañazos y manchas también se realizará una localización de donde esta situada la imperfección. Siendo por tanto la salida esperada que tiene que cumplir el sistema, la clasificación del tipo de imperfección que contiene la imagen y por otro lado la localización de las mismas. 

Para la realización del proyecto se tiene una base de datos en la que previamente se ha realizado una localización previa de la imperfección. Por tanto, el input recibido para el sistema son imagenes que acotan la pieza a la zona donde hay imperfección.

## Puesta en marcha

Para la ejecución de la aplicación bajo el sistema operativo Linux Ubuntu 20.04 LTS serán necesarios realizar estos pasos, teniendo instalado Python 3.8:

* Si no esta instalado git es necesario instalarlo y configurarlo, si no pasar al siguiente punto:
    * sudo apt-get install git
    * git config --global user.name "Your Name"
    * git config --global user.email "youremail@yourdomain.com"
* Clonar repositorio AIVA_2021_METAL_IMPERFECTIONS y entrar en la carpeta creada:
    * cd /path/to/save/repository (Ej. cd /home/user/)
    * git clone https://github.com/rubenOliver/AIVA_2021_METAL_IMPERFECTIONS.git
    * cd AIVA_2021_METAL_IMPERFECTIONS/
* Crear un entorno virtual y activarlo:
    * python3 -m venv ./venv
    * source ./venv/bin/activate
* Instalar dependecias del requirements.txt:
    * pip3 install -r requirements.txt
* Una vez estan las dependencias instaladas, lanzar la aplicación sobre la carpeta de imagenes:
    * python3 Application.py /path/to/images (Ej. python3 Application.py ./NEU-DET/IMAGES/)

## Despliegue mediante Docker
La imagen docker de la aplicación esta disponible en el siguente enlace: [Docker - bullseyemuva/aiva](https://hub.docker.com/r/bullseyemuva/aiva)

Para obtener la imagen docker hay dos opciones disponibles: 
* Para descargar y ejecutar la imagen docker con el mismo comando, serían los siguientes comandos:
    * En algunos casos, si no tenemos habilitado la conexión de docker con la pantalla de nuestra máquina es necesario realizar este comando: xhost +"local:docker@"
    * sudo docker run --rm -it -p 9000:9000 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix bullseyemuva/aiva
    * java -jar aiva.jar
    
* Descargar la imagen sin lanzar su ejecución en el momento, son los siguientes comandos:
    * sudo docker pull bullseyemuva/aiva
    * En algunos casos, si no tenemos habilitado la conexión de docker con la pantalla de nuestra máquina es necesario realizar este comando: xhost +"local:docker@"
    * sudo docker run --rm -it -p 9000:9000 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix bullseyemuva/aiva
    * java -jar aiva.jar

Con esto se despliega una interfaz de usuario que permite la selección de las imagenes de la base de datos para su procesamiento y análisis. 

Todo este procesamiento esta hecho dentro del docker, sin embargo, también existe la posibilidad de poder hacer una conexión externa al docker. Para ello será necesario realizar los siguientes pasos:
* Clonar el proyecto a nuestro sistema si no lo hemos hecho ya con el siguiente comando: git clone https://github.com/rubenOliver/AIVA_2021_METAL_IMPERFECTIONS.git
* Entrar en el proyecto clonado: cd AIVA_2021_METAL_IMPERFECTIONS
* Compilar las clases Java con los siguientes comandos:
    * javac -d ./classes -cp ./classes java-src-aiva/aiva/*.java
    * javac -d ./classes -cp ./classes java-src-aiva/*.java
* Copiar la carpeta MANIFEST a la localización de los compilados: cp -r META-INF/ ./classes/
* Movernos a la carpeta de compilados: cd ./classes
* Crear el jar en la raiz del proyecto: jar cmf META-INF/MANIFEST.MF ../aiva.jar *.class aiva
* Volver a la carpeta raiz del proyecto: cd ..
Llegados a este punto es necesario desplegar el docker en una nueva terminal, sin llegar a ejecutar el comando " java -jar aiva.jar "
Con el docker desplegado lanzamos el jar en nuestra máquina con el comando: java -jar aiva.jar 

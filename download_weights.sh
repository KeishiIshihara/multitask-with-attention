# https://drive.google.com/file/d/1J4GcBwFiL-d8UJwD8OvHffUAJtnFmPYx/view?usp=sharing
fileid="1J4GcBwFiL-d8UJwD8OvHffUAJtnFmPYx"
filename="ckpts.zip"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}
unzip ckpts.zip
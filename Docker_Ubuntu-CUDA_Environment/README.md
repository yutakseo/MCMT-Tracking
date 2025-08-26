## 💡 Motivation

While working on deep learning projects, I realized that **environment setup** is often the most frustrating and time-consuming part.  
To streamline this process, I built a ready-to-use **Docker environment** based on **Ubuntu 22.04** with **CUDA 11.8** support. This setup allows you to skip the hassle of **dependency conflicts** and focus directly on **model development and experimentation**.

 <br><br>

## Features
- ✅ One-command Docker setup with optional dataset mounting  
- ✅ Automatic image build if not found  
- ✅ Easy volume linking using `___DATASETS___.list`  
- ✅ Automatically links `requirements.txt` into container  
- ✅ CUDA 11.8 + Ubuntu 22.04 base for maximum compatibility  

 <br><br>

## Usage
### 1. Add your dataset path  

If you already have datasets stored on your machine, just write their paths in `___DATASETS___.list`.<br>
Edit the `___DATASETS___.list` file to include the absolute paths to your datasets(e.g., coco, VOC...), one per line.  

```bash
#[host]/___DATASETS___.list
/home/yourname/datasets/my_coco_dataset
/mnt/data/datasets/balloon_dataset
/mnt/data2/VOC_dataset
```

Each will be mounted inside the container under:

```bash
#[container]
/workspace/DATASETS/<dataset_name>
```


 <br>



### 2. Run the container  

```bash
#[host]
bash run.sh -v /path/to/your_project_dir
```

The `-v` option specifies the **host directory** to be mounted as the container’s working directory (`/workspace`).  
This is where your code and project files will be accessible inside the container.  

> If you omit the `-v` option, your **current working directory** will be mounted by default.



 <br>



### 3. Access the container  

Once the container is running, you can enter it using:

```bash
#[host]
docker exec -it ubuntu22.04_cuda11.08_container bash
```


<strong>Container Structure</strong>

```text
📁 /  # root
└── 📁 workspace
    ├── 📁 DATASETS
    │   ├── 📁 coco_example
    │   └── 📁 <another_dataset>
    │
    ├── 📁 <your_project_dir>   # e.g., Ultralytics, mmdetection
    └── 📄 requirements.txt     # symlinked automatically
```



- `/workspace` → your working directory  
- `/workspace/DATASETS/<dataset_name>` → dataset mounted via `___DATASETS___.list`  



<br>




### 4. Install pip dependencies
After entering the container, install Python dependencies using:
```bash
#[container]
pip install -r requirements.txt
```
This will install all the packages listed in your project's `requirements.txt`
The full list of dependencies is shown below.

<details>
<summary><strong> Python Dependencies (click to expand)</strong></summary>

```txt
# Use the PyTorch CUDA 11.8 wheel repository
--extra-index-url https://download.pytorch.org/whl/cu118

# Core DL packages with CUDA 11.8
torch==2.1.0+cu118
torchvision==0.16.0+cu118
torchaudio==2.1.0

# General dependencies
numpy==1.22.3
pillow==8.2.0
requests==2.32.3
certifi==2024.8.30
urllib3==2.2.3
idna==3.10
charset-normalizer==3.4.0
typing_extensions==4.12.2
pyyaml==6.0
filelock==3.16.1
jinja2==3.1.4
sympy==1.13.3
networkx==3.1
cffi==1.15.0
pycparser==2.22
pysocks==1.7.1
markupsafe==2.1.1
olefile==0.47
```

</details>


 <br><br>

### 5. Enjoy your deep learning development env! 😎

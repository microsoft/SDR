# The installations script can be executed as a bash script.
conda create -n SDR python=3.7 --yes
source ~/anaconda3/etc/profile.d/conda.sh
conda activate SDR

conda install -c pytorch pytorch==1.7.0 torchvision cudatoolkit=11.0 --yes 
pip install -U cython transformers==3.1.0 nltk pytorch-metric-learning joblib pytorch-lightning==1.1.8 pandas

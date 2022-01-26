try:
    !jupyter nbconvert --to python models.py
except:
    pass

from ipynb.fs.full.models import Res18Skip
print(Res18Baseline)

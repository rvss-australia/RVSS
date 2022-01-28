import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from PIL import Image

def arrowx(x,theta,size):
  c=np.cos(theta)
  s=np.sin(theta)
  return[x,
         x+c*size,
         x+(0.75*c-0.15*s)*size,
         None,
         x+c*size,
         x+(0.75*c+0.15*s)*size]

def arrowy(y,theta,size):
  s=np.sin(theta)
  c=np.cos(theta)
  return[y,
          y+s*size,
          y+(0.75*s+0.15*c)*size,
          None,
          y+s*size,
          y+(0.75*s-0.15*c)*size]

def robot(state):
  arrowsize=0.1
  rx=[]
  ry=[]
  ra=[]
  rc=[]
  for t in range(state.shape[0]):
    x=state[t,0]
    y=state[t,1]
    theta=state[t,2]
    rx+=arrowx(x,theta,arrowsize)
    ry+=arrowy(y,theta,arrowsize)
    ra+=[t]*6
    rc+=['r']*6
    rx+=arrowx(x,theta+np.pi/2,arrowsize)
    ry+=arrowy(y,theta+np.pi/2,arrowsize)
    ra+=[t]*6
    rc+=['g']*6
  return rx,ry,ra,rc

def Render(state):
  x,y,a,c = robot(state)

  colmap={'r':'red','g':'green'}
  fig = px.line(x=x,y=y,animation_frame=a, color=c, color_discrete_map=colmap)

  marker_files = [filename for filename in os.listdir('./image') if filename.startswith("M")]
  marker_world_width = 0.3
  for i,filename in enumerate(marker_files):
    fprts = filename.split('_')
    xpos = float(fprts[1])
    ypos = float(fprts[2])
    #mp = np.array([float(fprts[1]),float(fprts[2])])
    #mi = cv2.imread('./image/'+filename)
    im = Image.open('image/' + filename)
    imsize = 0.3
    # Add images
    fig.add_layout_image(
        dict(
            source = im,
            xref="x",
            yref="y",
            x=xpos-imsize/2,
            y=ypos+imsize/2,
            sizex=imsize,
            sizey=imsize,
            sizing="contain",
            opacity=1.0,
            layer="below")
    )

  fig.update_xaxes(showgrid=False, range = [-4,-2])
  fig.update_yaxes(showgrid=False, range = [-3.5,-1.5])
  fig.update_layout(template="plotly_white")
  fig.update_layout(showlegend=False)
  fig.update_layout(width = 500, height=530) #makes it look square on my screen
  fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 100
  fig.show()

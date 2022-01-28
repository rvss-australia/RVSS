import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from PIL import Image

def arrow(x,y,theta,size):
  c=np.cos(theta)
  s=np.sin(theta)
  return([x,
         x+c*size,
         x+(0.75*c-0.15*s)*size,
         None,
         x+c*size,
         x+(0.75*c+0.15*s)*size],
         [y,
          y+s*size,
          y+(0.75*s+0.15*c)*size,
          None,
          y+s*size,
          y+(0.75*s-0.15*c)*size])

def robot(state,xcol,ycol,rx,ry,ra,rc):
  arrowsize=0.1
  for t in range(state.shape[0]):
    x=state[t,0]
    y=state[t,1]
    theta=state[t,2]
    arrx,arry = arrow(x, y, theta, arrowsize)
    rx+=arrx
    ry+=arry
    ra+=[t]*6
    rc+=[xcol]*6
    arrx,arry = arrow(x, y, theta+np.pi/2, arrowsize)
    rx+=arrx
    ry+=arry
    ra+=[t]*6
    rc+=[ycol]*6

def ellipse(mm,numpts):
  a = mm.covariance[0,0]
  b = mm.covariance[0,1]
  c = mm.covariance[1,1]
  ap = 1/np.sqrt(a-b*b/c)
  bp = -b*ap/c
  cp = 1/np.sqrt(c)
  invsqrt=np.array([[a,0],[b,c]])
  xs,ys=[],[]
  for i in range(numpts+1):
    theta = i*2*np.pi/numpts
    coord = np.matmul(invsqrt,np.array([[np.cos(theta)],[np.sin[theta]]]))
    xs.append(coord[0,0]+mm.position[0])
    ys.append(coord[1,0]+mm.position[1])
  return (xs,ys)
  
def meas(m,mcol,rx,ry,ra,rc):
  for t in range(len(m)):
    mm=m[t]
    numpts=20
    xell,yell = ellipse(mm,numpts)
    rx+=ell
    ry+=yell
    ra+=[t]*(numpts+1)
    rc+=[mcol]*(numpts+1)

def Render(state=None, gtstate=None, measurements=None):
  x,y,a,c=[],[],[],[]

  if(state is not None):
    robot(state,'r','g',x,y,a,c)
    
  if(gtstate is not None):
    robot(gtstate,'dr','dg',x,y,a,c)

  if(measurements is not None):
    meas(measurements,'b',x,y,a,c)

  colmap={'r':'#FF0000','g':'#00FF00','dr':'#800000','dg':'#008000','b':'#8080ff'}
  fig = px.line(x=x,y=y,animation_frame=a, color=c, color_discrete_map=colmap)

  marker_files = [filename for filename in os.listdir('./image') if filename.startswith("M")]
  marker_world_width = 0.3
  for i,filename in enumerate(marker_files):
    fprts = filename.split('_')
    xpos = float(fprts[1])
    ypos = float(fprts[2])
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
  fig.update_layout(template="plotly_white", showlegend=False)
  fig.update_layout(width = 500, height=530) #makes it look square on my screen
  fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 100
  fig.show()

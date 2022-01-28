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

def ellipse(pos,covariance,numpts,scale=0.25):
  a = covariance[0,0]
  b = covariance[0,1]
  c = covariance[1,1]
  cp = np.sqrt(c)
  bp = b/cp
  ap = np.sqrt(a-bp*bp)
  sqrt=np.array([[ap,0],[bp,cp]])
  xs,ys=[],[]
  for i in range(numpts+1):
    theta = i*2*np.pi/numpts
    circlepos = np.array([scale*np.cos(theta),scale*np.sin(theta)])
    coord = np.matmul(sqrt,circlepos)+pos
    xs.append(coord[0])
    ys.append(coord[1])
  xs.append(None)
  ys.append(None)  
  return (xs,ys)
  
def meas(state,m,mcol,rx,ry,ra,rc):
  for t in range(len(m)):
    rpos = state[t,0:2]
    rtheta = state[t,2]
    c=np.cos(rtheta)
    s=np.sin(rtheta)
    rrot = np.array([[c,-s],[s,c]])

    mms=m[t]
    numpts=20
    for mm in mms:
      mpos = mm.position[:,0]
      mcov = mm.covariance

      mwpos = rpos+np.matmul(rrot,mpos)
      mwcov = np.matmul(np.matmul(rrot,mcov),rrot.transpose())

      xell,yell = ellipse(mwpos,mwcov,numpts)
      rx+=xell
      ry+=yell
      ra+=[t]*len(xell)
      rc+=[mcol]*len(xell)

def drawrcov(state, rcov, col, rx, ry, ra, rc):
  for t in range(len(state)):
    rpos = state[t,0:2]
    cov = rcov[t]
    xs, ys = ellipse(rpos, cov, 20)
    rx+=xs
    ry+=ys
    ra+=[t]*len(xs)
    rc+=[col]*len(xs)


def Render(state=None, gtstate=None, measurements=None, rcov=None):
  x,y,a,c=[],[],[],[]

  if(state is not None):
    robot(state,'r','g',x,y,a,c)
    
  if(gtstate is not None):
    robot(gtstate,'dr','dg',x,y,a,c)

  if(measurements is not None):
    meas(state, measurements,'b',x,y,a,c)

  if(rcov is not None):
    drawrcov(state,rcov,'c',x,y,a,c)

  colmap={'r':'#FF0000','g':'#00FF00','dr':'#800000','dg':'#008000','b':'#8080ff', 'c':'#00ffff'}
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

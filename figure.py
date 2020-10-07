import numpy as np
from pathlib import Path

from plotly.subplots import make_subplots
import plotly.graph_objs as go

import fym.logging

import config as cfg


datapath = Path("data")
envpathlist = [p for p in datapath.iterdir()]
r2d = np.rad2deg(1)

# Figure 1
fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02)

style = dict(
    line=[
        dict(color="red", dash="solid"),
        dict(color="green", dash="solid"),
        dict(color="blue", dash="solid"),
    ],
    reference=dict(color="black", dash="dash"),
    command=dict(color="black"),
    true_param=dict(color="red", dash="dash"),
)

for envpath, line_style in zip(envpathlist, style["line"]):
    trajdata, info = fym.logging.load(envpath / "traj.h5", with_info=True)
    agentdata = fym.logging.load(envpath / "agent.h5")

    name = info["name"]
    t = trajdata["t"]
    xp = trajdata["state"]["system"]["x"]["xp"]
    xr = trajdata["state"]["system"]["xr"]
    rint = trajdata["state"]["rint"]
    u = trajdata["u"]
    zcmd = trajdata["zcmd"]

    # Row 1
    fig.add_trace(
        row=1, col=1,
        trace=go.Scatter(
            x=t,
            y=xp[:, 0, 0] * r2d,
            line=line_style,
            legendgroup=name,
            name=name,
        ),
    )

    # Row 2
    fig.add_trace(
        row=2, col=1,
        trace=go.Scatter(
            x=t,
            y=xp[:, 1, 0] * r2d,
            line=line_style,
            legendgroup=name,
            showlegend=False,
        ),
    )

    # Row 3
    fig.add_trace(
        row=3, col=1,
        trace=go.Scatter(
            x=t,
            y=u[:, 0, 0],
            line=line_style,
            legendgroup=name,
            showlegend=False,
        ),
    )

    # Row 4
    fig.add_trace(
        row=4, col=1,
        trace=go.Scatter(
            x=t,
            y=rint[:, 0, 0],
            line=line_style,
            legendgroup=name,
            showlegend=False,
        ),
    )

    if name in ["MRAC", "CMRAC"]:
        What = trajdata["state"]["controller"]["What"].squeeze()

        for w in What.T:
            # Row 5
            fig.add_trace(
                row=5, col=1,
                trace=go.Scatter(
                    x=t,
                    y=w,
                    line=line_style,
                    legendgroup=name,
                    showlegend=False
                )
            )

# References and commands
fig.add_traces(
    rows=1, cols=1,
    data=[
        go.Scatter(
            x=t,
            y=xr[:, 0, 0] * r2d,
            line=style["reference"],
            legendgroup="reference",
            name="Reference",
        ),
        go.Scatter(
            x=t,
            y=zcmd * r2d,
            line=style["command"],
            legendgroup="command",
            name="Command",
        ),
    ],
)

fig.add_traces(
    rows=2, cols=1,
    data=[
        go.Scatter(
            x=t,
            y=xr[:, 1, 0] * r2d,
            line=style["reference"],
            legendgroup="reference",
            showlegend=False,
        ),
    ],
)

shapes = []
for w in cfg.PLANT.Wast.ravel():
    shapes.append(
        dict(
            type="line",
            xref="paper",
            yref="y5",
            x0=0,
            x1=1,
            y0=w,
            y1=w,
            line=style["true_param"],
        )
    )

fig.update_layout(shapes=shapes)


fig.update_yaxes(row=1, col=1,
                 title_text=r"$\phi \text{ (deg)}$",
                 range=[-80, 80])
fig.update_yaxes(row=2, col=1,
                 title_text=r"$\dot{\phi} \text{ (deg/s)}$",
                 range=[-80, 80])
fig.update_yaxes(row=3, col=1,
                 title_text=r"$\delta_e \text{ (deg)}$",
                 range=[-50, 50])
fig.update_yaxes(row=4, col=1,
                 title_text=r"$J$")
fig.update_yaxes(row=5, col=1,
                 title_text=r"$\hat{W}$")
fig.update_yaxes(row=6, col=1,
                 title_text=r"$\theta$")
fig.update_xaxes(row=6, col=1,
                 title_text=r"$\text{Time (sec)}$")

fig.update_layout(
    width=800,
    height=1000,
    showlegend=True,
)

fig.show()

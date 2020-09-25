import numpy as np
from pathlib import Path

from plotly.subplots import make_subplots
import plotly.graph_objs as go

import fym.logging


datapath = Path("Data")
runpaths = sorted(datapath.glob("run-*"))

traj_dataset = {}
for p in runpaths:
    data, info = fym.logging.load(p / "traj.h5", with_info=True)
    traj_dataset[p.name] = dict(data=data, info=info)

agent_dataset = {p.name: fym.logging.load(p / "agent.h5") for p in runpaths}

traj_data = traj_dataset["run-000"]["data"]
traj_info = traj_dataset["run-000"]["info"]
agent_data = agent_dataset["run-000"]

r2d = np.rad2deg(1)

# # Figure 1
# fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)

# for k, traj in traj_dataset.items():
#     traj_data = traj["data"]
#     fig.add_trace(
#         go.Scatter(
#             x=traj_data["t"],
#             y=traj_data["x"][:, 0, 0] * r2d,
#         ),
#         row=1, col=1,
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=traj_data["t"],
#             y=traj_data["x"][:, 1, 0] * r2d,
#         ),
#         row=2, col=1,
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=traj_data["t"],
#             y=traj_data["u"][:, 0, 0] * r2d,
#         ),
#         row=3, col=1,
#     )

# fig.update_yaxes(title_text=r"$\phi$ (deg)", row=1, col=1)
# fig.update_yaxes(title_text=r"$\dot{\phi}$ (deg/s)", row=2, col=1)
# fig.update_yaxes(title_text=r"$\delta_e$ (deg)", row=3, col=1)

# fig.show()

# # Figure 2
# fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)

# for k, agent_data in agent_dataset.items():
#     fig.add_trace(
#         go.Scatter(
#             x=agent_data["t"],
#             y=agent_data["info"]["HJB_error"],
#         ),
#         row=1, col=1
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=agent_data["t"],
#             y=agent_data["info"]["actor_error"],
#         ),
#         row=2, col=1
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=agent_data["t"],
#             y=agent_data["info"]["actor_param"][:, 0, 0],
#         ),
#         row=3, col=1
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=agent_data["t"],
#             y=agent_data["info"]["actor_param"][:, 1, 0],
#         ),
#         row=3, col=1
#     )

# fig.add_shape(
#     type="line",
#     x0=0,
#     x1=agent_data["t"].max(),
#     y0=traj_info["optimal_gain"][0, 0],
#     y1=traj_info["optimal_gain"][0, 0],
#     row=3, col=1
# )
# fig.add_shape(
#     type="line",
#     x0=0,
#     x1=agent_data["t"].max(),
#     y0=traj_info["optimal_gain"][0, 1],
#     y1=traj_info["optimal_gain"][0, 1],
#     row=3, col=1
# )

# fig.show()

# Figure 3
fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02)

for i, ((traj_key, traj), (agent_key, agent_data)) in enumerate(zip(traj_dataset.items(), agent_dataset.items())):
    traj_data = traj["data"]

    tmod = i * 20

    fig.add_trace(
        go.Scatter(
            x=traj_data["t"] + tmod,
            y=traj_data["x"][:, 0, 0] * r2d,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=traj_data["t"] + tmod,
            y=traj_data["x"][:, 1, 0] * r2d,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=traj_data["t"] + tmod,
            y=traj_data["u"][:, 0, 0] * r2d,
        ),
        row=3, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=agent_data["t"] + tmod,
            y=agent_data["info"]["HJB_error"],
        ),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=agent_data["t"] + tmod,
            y=agent_data["info"]["actor_error"],
        ),
        row=5, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=agent_data["t"] + tmod,
            y=agent_data["info"]["actor_param"][:, 0, 0],
        ),
        row=6, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=agent_data["t"] + tmod,
            y=agent_data["info"]["actor_param"][:, 1, 0],
        ),
        row=6, col=1
    )

fig.add_shape(
    type="line",
    x0=0,
    x1=(agent_data["t"] + tmod).max(),
    y0=traj_info["optimal_gain"][0, 0],
    y1=traj_info["optimal_gain"][0, 0],
    row=6, col=1
)
fig.add_shape(
    type="line",
    x0=0,
    x1=(agent_data["t"] + tmod).max(),
    y0=traj_info["optimal_gain"][0, 1],
    y1=traj_info["optimal_gain"][0, 1],
    row=6, col=1
)

fig.update_yaxes(title_text=r"$\phi \text{ (deg)}$", row=1, col=1)
fig.update_yaxes(title_text=r"$\dot{\phi} \text{ (deg/s)}$", row=2, col=1)
fig.update_yaxes(title_text=r"$\delta_e \text{ (deg)}$", row=3, col=1)
fig.update_yaxes(title_text=r"$e_{HJB}$", row=4, col=1)
fig.update_yaxes(title_text=r"$e_a$", row=5, col=1)
fig.update_yaxes(title_text=r"$\theta$", row=6, col=1)
fig.update_xaxes(title_text=r"$\text{Time (sec)}$", row=6, col=1)

fig.update_layout(
    width=800,
    height=1000,
    showlegend=False,
)

fig.show()

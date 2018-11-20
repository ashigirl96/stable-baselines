import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row, widgetbox, Spacer
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from stable_baselines.common.cmd_util import make_atari_env

env_id = 'BreakoutNoFrameskip-v4'
env = make_atari_env(env_id, 1, 0)

print(env.reset()[0, ..., 0].shape)

observs = [env.reset()[0, ..., 0]]

num_timesteps = 100
for _ in range(num_timesteps - 1):
  actions = [env.action_space.sample()]
  observ, rewards, terminals, infos = env.step(actions)
  observs.append(observ[0, ..., 0])

coef = 5
x = np.linspace(0, num_timesteps, num=num_timesteps * coef)
y = 5 * np.sin(x)
source = ColumnDataSource(data=dict(image=[observs[0]]))
line_source = ColumnDataSource(data=dict(x=x, y=y))
circle_source = ColumnDataSource(data=dict(x=[x[0]], y=[y[0]]))

x_max = env.observation_space.shape[0]
y_max = env.observation_space.shape[1]
plot = figure(x_range=(0, x_max), y_range=(0, y_max), plot_width=500, plot_height=500)
plot.image(image='image', x=0, y=0, dw=x_max, dh=y_max, source=source, palette="Spectral11")

TOOLTIPS = [
  ("TImeStep", "@x"),
  ("Successor Feature Value", "@y")
]

line_plot = figure(x_range=(0, num_timesteps), y_range=(-10, 10),
                   plot_width=1000, plot_height=300, tooltips=TOOLTIPS,
                   title="Successor Features")
line_plot.line('x', 'y', source=line_source, line_width=3, line_alpha=0.6)
line_plot.circle(x="x", y="y", source=circle_source, size=15,
                 color="navy", alpha=0.5)

frame_slider = Slider(start=0, end=(num_timesteps - 1), value=0, step=1, title="Frame", )


def update(attr, old, new):
  source.data = dict(image=[observs[frame_slider.value]])
  line_source.data = dict(x=x, y=y)
  circle_source.data = dict(x=[x[frame_slider.value * coef]],
                            y=[y[frame_slider.value * coef]])


frame_slider.on_change('value', update)

layout = row(
  plot,
  column(
    widgetbox(frame_slider),
    Spacer(height=20),
    line_plot,
  )
)

curdoc().add_root(layout)

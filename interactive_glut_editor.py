"""
GLUT Interactive Editor
=======================
Dash application for editing a trained Gaussian LUT by specifying
input → output color constraints directly on an exemplar image.

Usage:
    python edit_glut_app.py \
        --pretrained_model path/to/model.pth \
        --image             path/to/image.jpg   # optional — can upload in GUI
        --num_gaussians     32

Workflow:
    1. Upload an image via the toolbar button (or pass --image at launch).
    2. Click a pixel on the original image  → sets x_e (blue marker).
    3. Pick the desired output color with the color picker → sets t.
    4. Tune K and Strength sliders.
    5. Press "Apply Edit" → GLUT is updated in-place.
    6. The edited image, 3D cube, and delta badge refresh instantly.
    7. Press "Download All" to get original / default / edited as a ZIP.
    8. Press "Save Model" to write the edited checkpoint.
    9. Press "Reset" to restore the original parameters.
"""

import os
import io
import base64
import zipfile
import argparse

import numpy as np
import torch
import plotly.graph_objects as go
from PIL import Image

import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc


# ---------------------------------------------------------------------------
# GLUT editor logic
# ---------------------------------------------------------------------------

class GLUTEditor:
    def __init__(self, model, device):
        self.model  = model
        self.device = device
        self._snapshot_params()

    def _snapshot_params(self):
        with torch.no_grad():
            self._original_biases = self.model.color_biases.detach().clone()

    def reset(self):
        with torch.no_grad():
            self.model.color_biases.copy_(self._original_biases)

    def _get_params(self):
        with torch.no_grad():
            pos = self.model.positions.detach().cpu().numpy()
            cov = self.model.get_covariance_matrix().detach().cpu().numpy()
            op  = torch.sigmoid(self.model.opacities_logit).detach().cpu().numpy().flatten()
        return pos, cov, op

    def _compute_weights(self, x: np.ndarray) -> np.ndarray:
        pos, cov, op = self._get_params()
        N = len(pos)
        densities = np.zeros(N)
        for i in range(N):
            diff = x - pos[i]
            try:
                cov_inv = np.linalg.inv(cov[i])
            except np.linalg.LinAlgError:
                cov_inv = np.eye(3)
            det   = max(np.linalg.det(cov[i]), 1e-12)
            mahal = diff @ cov_inv @ diff
            norm  = 1.0 / ((2 * np.pi) ** 1.5 * det ** 0.5)
            densities[i] = norm * np.exp(-0.5 * mahal)
        w = densities * op
        return w / (w.sum() + 1e-8)

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
            yt = self.model(xt)
        return np.clip(yt.squeeze(0).cpu().numpy(), 0.0, 1.0)

    def apply_to_image(self, img_np: np.ndarray) -> np.ndarray:
        h, w, _ = img_np.shape
        flat = img_np.reshape(-1, 3)
        with torch.no_grad():
            t = torch.tensor(flat, dtype=torch.float32, device=self.device)
            out = self.model(t).cpu().numpy()
        return np.clip(out.reshape(h, w, 3), 0.0, 1.0)

    def edit_single(self, x_in, x_out, K=5, strength=1.0):
        x_in  = np.clip(np.array(x_in,  dtype=float), 0, 1)
        x_out = np.clip(np.array(x_out, dtype=float), 0, 1)

        y_cur = self.forward_numpy(x_in)
        delta = x_out - y_cur

        if np.linalg.norm(delta) < 1e-5:
            return {'status': 'already satisfied', 'error_before': 0.0}

        error_before = float(np.linalg.norm(delta) * 255)

        weights  = self._compute_weights(x_in)
        K        = min(K, len(weights))
        topk_idx = np.argsort(weights)[-K:]
        topk_w   = weights[topk_idx]
        alpha    = topk_w / topk_w.sum()

        with torch.no_grad():
            for k, a in zip(topk_idx, alpha):
                db = strength * a * delta
                self.model.color_biases[k] += torch.tensor(
                    db, dtype=torch.float32, device=self.device)

        y_after = self.forward_numpy(x_in)
        error_after = float(np.linalg.norm(x_out - y_after) * 255)

        return {
            'status'      : 'ok',
            'error_before': error_before,
            'error_after' : error_after,
            'topk_indices': topk_idx.tolist(),
            'alpha'       : alpha.tolist(),
        }

    def get_gaussian_traces(self, constraints=None):
        pos, cov, op = self._get_params()
        traces = []

        for i in range(len(pos)):
            if op[i] < 0.02:
                continue
            center  = pos[i]
            eigvals, eigvecs = np.linalg.eigh(cov[i])
            eigvals = np.maximum(eigvals, 1e-6)
            axes    = np.sqrt(eigvals)

            u  = np.linspace(0, 2 * np.pi, 18)
            v  = np.linspace(0, np.pi,     18)
            ug, vg = np.meshgrid(u, v)
            xs = axes[0] * np.cos(ug) * np.sin(vg)
            ys = axes[1] * np.sin(ug) * np.sin(vg)
            zs = axes[2] * np.cos(vg)
            pts = eigvecs @ np.stack([xs.ravel(), ys.ravel(), zs.ravel()])
            X = pts[0].reshape(xs.shape) + center[0]
            Y = pts[1].reshape(ys.shape) + center[1]
            Z = pts[2].reshape(zs.shape) + center[2]

            o = float(op[i])
            r_int = int(np.clip(center[0] * 255, 0, 255))
            g_int = int(np.clip(center[1] * 255, 0, 255))
            b_int = int(np.clip(center[2] * 255, 0, 255))
            rgb = f"{r_int},{g_int},{b_int}"
            traces.append(go.Surface(
                x=X, y=Y, z=Z,
                opacity=o * 0.25,
                colorscale=[[0, f'rgba({rgb},{o:.2f})'],
                            [1, f'rgba({rgb},{o:.2f})']],
                showscale=False,
                name=f'G{i}',
                hovertemplate=(
                    f'<b>Gaussian {i}</b><br>'
                    f'µ = ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})<br>'
                    f'opacity = {o:.3f}<extra></extra>'
                )
            ))
            traces.append(go.Scatter3d(
                x=[center[0]], y=[center[1]], z=[center[2]],
                mode='markers',
                marker=dict(size=3 + 4 * o,
                            color=f'rgb({rgb})',
                            line=dict(color='white', width=1)),
                showlegend=False,
                hoverinfo='skip'
            ))

        edges = [
            [[0,0,0],[1,0,0]], [[0,0,0],[0,1,0]], [[0,0,0],[0,0,1]],
            [[1,0,0],[1,1,0]], [[1,0,0],[1,0,1]],
            [[0,1,0],[1,1,0]], [[0,1,0],[0,1,1]],
            [[0,0,1],[1,0,1]], [[0,0,1],[0,1,1]],
            [[1,1,0],[1,1,1]], [[1,0,1],[1,1,1]], [[0,1,1],[1,1,1]]
        ]
        for e in edges:
            traces.append(go.Scatter3d(
                x=[e[0][0], e[1][0]], y=[e[0][1], e[1][1]], z=[e[0][2], e[1][2]],
                mode='lines', line=dict(color='rgba(255,255,255,0.15)', width=1),
                showlegend=False, hoverinfo='none'
            ))

        if constraints:
            for xi, xo in constraints:
                xi, xo = np.array(xi), np.array(xo)
                traces.append(go.Scatter3d(
                    x=[xi[0]], y=[xi[1]], z=[xi[2]],
                    mode='markers',
                    marker=dict(size=10, color='#4fc3f7', symbol='circle'),
                    name='x_e', hoverinfo='text',
                    text=f'x_e: {(xi*255).astype(int)}'
                ))
                traces.append(go.Scatter3d(
                    x=[xo[0]], y=[xo[1]], z=[xo[2]],
                    mode='markers',
                    marker=dict(size=10, color='#81c784', symbol='diamond'),
                    name='t', hoverinfo='text',
                    text=f't: {(xo*255).astype(int)}'
                ))
                traces.append(go.Scatter3d(
                    x=[xi[0], xo[0]], y=[xi[1], xo[1]], z=[xi[2], xo[2]],
                    mode='lines', line=dict(color='#ffb74d', width=5),
                    showlegend=False, hoverinfo='none'
                ))

        return traces


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def np_to_b64(arr: np.ndarray) -> str:
    img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def b64_to_np(contents: str, max_size: int = 512) -> np.ndarray:
    """Decode a dcc.Upload data-URL to a float32 HxWx3 numpy array."""
    _header, data = contents.split(',', 1)
    buf = io.BytesIO(base64.b64decode(data))
    img = Image.open(buf).convert('RGB')
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def load_image(path: str, max_size: int = 512) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def build_cube_figure(editor, constraints=None):
    traces = editor.get_gaussian_traces(constraints)
    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor='#0d0d0d',
        plot_bgcolor ='#0d0d0d',
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(title='R', range=[0,1],
                       backgroundcolor='#111', gridcolor='#222',
                       showbackground=True, tickfont=dict(color='#666')),
            yaxis=dict(title='G', range=[0,1],
                       backgroundcolor='#111', gridcolor='#222',
                       showbackground=True, tickfont=dict(color='#666')),
            zaxis=dict(title='B', range=[0,1],
                       backgroundcolor='#111', gridcolor='#222',
                       showbackground=True, tickfont=dict(color='#666')),
            aspectmode='cube',
            bgcolor='#0d0d0d',
        ),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#aaa')),
    )
    return fig


def _image_figure(b64_src: str, fig_id: str, img_hw=(512, 512)):
    h, w = img_hw
    fig = go.Figure()

    fig.add_layout_image(
        source=b64_src,
        xref='x', yref='y',
        x=0, y=h,
        sizex=w, sizey=h,
        sizing='stretch',
        layer='below'
    )

    if fig_id == 'original':
        step = max(1, w // 64)
        xs = list(range(0, w, step))
        ys = list(range(0, h, step))
        xg, yg = np.meshgrid(xs, ys)
        fig.add_trace(go.Scatter(
            x=xg.ravel().tolist(),
            y=yg.ravel().tolist(),
            mode='markers',
            marker=dict(size=step, opacity=0, color='rgba(0,0,0,0)'),
            hoverinfo='none',
            showlegend=False,
        ))

    fig.update_xaxes(showticklabels=False, showgrid=False,
                     zeroline=False, range=[0, w], constrain='domain')
    fig.update_yaxes(showticklabels=False, showgrid=False,
                     zeroline=False, range=[0, h], scaleanchor='x')
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#161616',
        plot_bgcolor='#161616',
        dragmode='zoom',
        clickmode='event',
    )
    return fig


def _btn_style(bg, fg):
    return {
        'background'   : bg,
        'color'        : fg,
        'border'       : '1px solid #2a2a2a',
        'borderRadius' : '6px',
        'padding'      : '8px 16px',
        'cursor'       : 'pointer',
        'fontSize'     : '12px',
        'fontFamily'   : "'DM Mono', monospace",
        'letterSpacing': '0.08em',
        'fontWeight'   : '600',
        'transition'   : 'opacity 0.15s',
        'whiteSpace'   : 'nowrap',
    }


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

def build_app(editor: GLUTEditor, img_np: np.ndarray, model_path: str,
              models_dir: str = 'models', num_gaussians: int = 32, residual: bool = True):

    # Mutable container so callbacks can swap the image in-place
    _img = [img_np]   # _img[0] is always the current image array

    # List available models in the models folder
    model_options = sorted([f for f in os.listdir(models_dir) if f.endswith('.pth')])

    default_glut_b64 = np_to_b64(editor.apply_to_image(_img[0]))

    DARK   = '#0d0d0d'
    PANEL  = '#161616'
    BORDER = '#2a2a2a'
    ACCENT = '#4fc3f7'
    GREEN  = '#81c784'
    AMBER  = '#ffb74d'
    TEXT   = '#ffffff'
    MUTED  = '#666'

    label_style = dict(color=TEXT, fontSize='12px',
                       letterSpacing='0.08em', textTransform='uppercase',
                       marginBottom='4px')
    swatch_style = lambda color: {
        'width': '36px', 'height': '36px', 'borderRadius': '6px',
        'border': f'2px solid {BORDER}',
        'background': color if color else '#222',
        'display': 'inline-block', 'verticalAlign': 'middle',
        'marginRight': '8px'
    }

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title='GLUT Editor'
    )

    app.layout = html.Div(style={
        'background': DARK, 'minHeight': '100vh',
        'fontFamily': "'DM Mono', 'Courier New', monospace",
        'color': TEXT, 'padding': '24px'
    }, children=[

        # ── title bar ─────────────────────────────────────────────────────
        html.Div(style={
            'marginBottom': '20px', 'borderBottom': f'1px solid {BORDER}',
            'paddingBottom': '16px',
            'display': 'flex', 'alignItems': 'center',
            'justifyContent': 'space-between',
        }, children=[
            # left: title
            html.Div(style={'display': 'flex', 'alignItems': 'baseline',
                            'gap': '16px'}, children=[
                html.Span('GLUT', style={'fontSize': '22px', 'fontWeight': '700',
                                         'color': ACCENT, 'letterSpacing': '0.15em'}),
                html.Span('EDITOR', style={'fontSize': '22px', 'fontWeight': '300',
                                           'letterSpacing': '0.15em'}),
                html.Span(f'· {os.path.basename(model_path)}',
                          style={'fontSize': '12px', 'color': TEXT}),
            ]),
            # right: upload + download
            html.Div(style={'display': 'flex', 'alignItems': 'center',
                            'gap': '10px'}, children=[
                dcc.Upload(
                    id='upload-image',
                    accept='image/*',
                    multiple=False,
                    children=html.Button(
                        '⬆  LOAD IMAGE', id='btn-upload',
                        style=_btn_style('#1a1a2e', ACCENT)
                    ),
                ),
                html.Button('⬇  DOWNLOAD ALL', id='btn-download-all',
                            n_clicks=0,
                            style=_btn_style('#1a2a1a', GREEN)),
                dcc.Download(id='download-zip'),
            ]),
        ]),

        # ── main grid ─────────────────────────────────────────────────────
        html.Div(style={'display': 'grid',
                        'gridTemplateColumns': '1fr 1fr 1fr 320px',
                        'gap': '16px', 'alignItems': 'start'}, children=[

            # col 1: original  (hover shows a magnifier loupe for precise picking)
            html.Div(style={'background': PANEL, 'borderRadius': '8px',
                            'border': f'1px solid {BORDER}',
                            'overflow': 'hidden',
                            'position': 'relative'}, children=[
                html.Div('ORIGINAL — click to select color',
                         style={**label_style, 'padding': '10px 14px'}),
                dcc.Graph(id='img-original', config={'displayModeBar': False},
                          style={'height': '380px'},
                          figure=_image_figure(np_to_b64(_img[0]), 'original',
                                               img_hw=_img[0].shape[:2])),
                # magnifier overlay - sized by the canvas it holds
                html.Div(id='magnifier', style={
                    'position': 'absolute',
                    'borderRadius': '50%',
                    'border': f'3px solid {ACCENT}',
                    'pointerEvents': 'none',
                    'display': 'none',
                    'overflow': 'hidden',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.6)',
                    'zIndex': '999',
                }),
            ]),

            # col 2: default GLUT (frozen per image)
            html.Div(style={'background': PANEL, 'borderRadius': '8px',
                            'border': f'1px solid {BORDER}',
                            'overflow': 'hidden'}, children=[
                html.Div('DEFAULT GLUT OUTPUT',
                         style={**label_style, 'padding': '10px 14px'}),
                dcc.Graph(id='img-default', config={'displayModeBar': False},
                          style={'height': '380px'},
                          figure=_image_figure(default_glut_b64, 'default',
                                               img_hw=_img[0].shape[:2]))
            ]),

            # col 3: edited output
            html.Div(style={'background': PANEL, 'borderRadius': '8px',
                            'border': f'1px solid {BORDER}',
                            'overflow': 'hidden'}, children=[
                html.Div('EDITED OUTPUT',
                         style={**label_style, 'padding': '10px 14px'}),
                dcc.Graph(id='img-edited', config={'displayModeBar': False},
                          style={'height': '380px'},
                          figure=_image_figure(default_glut_b64, 'edited',
                                               img_hw=_img[0].shape[:2]))
            ]),

            # col 4: controls
            html.Div(style={'background': PANEL, 'borderRadius': '8px',
                            'border': f'1px solid {BORDER}', 'padding': '16px',
                            'display': 'flex', 'flexDirection': 'column',
                            'gap': '18px'}, children=[

                html.Div([
                    html.Div('INPUT COLOR  (c_in)', style=label_style),
                    # swatch + current value label
                    html.Div(style={'display': 'flex', 'alignItems': 'center',
                                    'gap': '10px', 'marginBottom': '8px'}, children=[
                        html.Div(id='swatch-in', style=swatch_style(None)),
                        html.Span(id='label-in', children='—',
                                  style={'fontSize': '13px', 'color': TEXT,
                                         'fontFamily': 'monospace'}),
                    ]),
                    # manual RGB entry
                    html.Div(style={'display': 'flex', 'gap': '6px',
                                    'alignItems': 'center'}, children=[
                        dbc.Input(id='input-r', type='number', placeholder='R',
                                  min=0, max=255, step=1,
                                  style={'width': '60px', 'height': '30px',
                                         'background': '#111', 'color': TEXT,
                                         'border': f'1px solid {BORDER}',
                                         'borderRadius': '4px', 'fontSize': '12px',
                                         'textAlign': 'center', 'padding': '2px 4px'}),
                        dbc.Input(id='input-g', type='number', placeholder='G',
                                  min=0, max=255, step=1,
                                  style={'width': '60px', 'height': '30px',
                                         'background': '#111', 'color': TEXT,
                                         'border': f'1px solid {BORDER}',
                                         'borderRadius': '4px', 'fontSize': '12px',
                                         'textAlign': 'center', 'padding': '2px 4px'}),
                        dbc.Input(id='input-b', type='number', placeholder='B',
                                  min=0, max=255, step=1,
                                  style={'width': '60px', 'height': '30px',
                                         'background': '#111', 'color': TEXT,
                                         'border': f'1px solid {BORDER}',
                                         'borderRadius': '4px', 'fontSize': '12px',
                                         'textAlign': 'center', 'padding': '2px 4px'}),
                        html.Button('SET', id='btn-set-xe', n_clicks=0,
                                    style={**_btn_style('#222', ACCENT),
                                           'padding': '4px 10px',
                                           'fontSize': '11px', 'height': '30px'}),
                    ]),
                ]),

                html.Div([
                    html.Div('TARGET COLOR  (c_out)', style=label_style),
                    dbc.Input(type='color', id='picker-out', value='#ffffff',
                              style={'width': '100%', 'height': '44px',
                                     'border': f'1px solid {BORDER}',
                                     'borderRadius': '6px', 'background': '#111',
                                     'cursor': 'pointer'}),
                    html.Div(id='label-out', children='255, 255, 255',
                             style={'fontSize': '13px', 'color': TEXT,
                                    'fontFamily': 'monospace', 'marginTop': '4px'}),
                ]),

                html.Div([
                    html.Div(id='label-K', children='K GAUSSIANS  ·  5',
                             style=label_style),
                    dcc.Slider(id='slider-K', min=1, max=16, step=1, value=5,
                               marks={1:'1', 4:'4', 8:'8', 12:'12', 16:'16'},
                               tooltip={'always_visible': False}),
                ]),

                html.Div([
                    html.Div(id='label-str', children='STRENGTH  ·  1.00',
                             style=label_style),
                    dcc.Slider(id='slider-str', min=0.0, max=1.0,
                               step=0.05, value=1.0,
                               marks={0:'0', 0.5:'0.5', 1:'1'},
                               tooltip={'always_visible': False}),
                ]),

                html.Div(style={'display': 'flex', 'flexDirection': 'column',
                                'gap': '8px'}, children=[
                    html.Button('▶  APPLY EDIT', id='btn-apply', n_clicks=0,
                                style={**_btn_style(ACCENT, '#0d0d0d'),
                                       'width': '100%'}),
                    html.Button('↺  RESET', id='btn-reset', n_clicks=0,
                                style={**_btn_style(BORDER, TEXT),
                                       'width': '100%'}),
                    html.Button('⬇  SAVE MODEL', id='btn-save', n_clicks=0,
                                style={**_btn_style('#1a2a1a', GREEN),
                                       'width': '100%'}),
                ]),

                html.Div(id='status-box',
                         style={'background': '#111', 'borderRadius': '6px',
                                'border': f'1px solid {BORDER}',
                                'padding': '10px 12px',
                                'fontSize': '11px', 'color': MUTED,
                                'minHeight': '56px'}),
            ]),
        ]),

        # ── Model selector (below images) ─────────────────────────────────
        html.Div(style={'display': 'grid',
                        'gridTemplateColumns': '1fr 1fr 1fr 320px',
                        'gap': '16px', 'marginTop': '16px'}, children=[
            html.Div(style={'background': PANEL, 'borderRadius': '8px',
                            'border': f'1px solid {BORDER}',
                            'padding': '12px 14px',
                            'gridColumn': '1 / 4',
                            'display': 'flex', 'alignItems': 'center', 'gap': '16px'}, children=[
                html.Div('MODEL', style={**label_style, 'marginBottom': '0'}),
                dcc.Dropdown(
                    id='dropdown-model',
                    options=[{'label': m, 'value': m} for m in model_options],
                    value=os.path.basename(model_path),
                    clearable=False,
                    style={'flex': '1', 'background': '#111', 'color': '#000'},
                ),
            ]),
        ]),

        # ── RGB cube (below images, aligned with first 3 columns) ─────────
        html.Div(style={'display': 'grid',
                        'gridTemplateColumns': '1fr 1fr 1fr 320px',
                        'gap': '16px', 'marginTop': '16px'}, children=[
            html.Div(style={'background': PANEL, 'borderRadius': '8px',
                            'border': f'1px solid {BORDER}',
                            'overflow': 'hidden',
                            'gridColumn': '1 / 4'}, children=[
                html.Div('GAUSSIAN RGB CUBE',
                         style={**label_style, 'padding': '10px 14px'}),
                dcc.Graph(id='cube-graph', style={'height': '460px'},
                          figure=build_cube_figure(editor),
                          config={'displayModeBar': True}),
            ]),
        ]),

        # ── hidden stores ─────────────────────────────────────────────────
        dcc.Store(id='store-constraint', data={'x_in': None}),
        # current original-image b64, fed to the magnifier's clientside callback
        dcc.Store(id='store-img-b64-js', data=np_to_b64(_img[0])),
        # b64 of the three panels, kept in sync for download
        dcc.Store(id='store-original-b64', data=np_to_b64(_img[0])),
        dcc.Store(id='store-default-b64',  data=default_glut_b64),
        dcc.Store(id='store-edited-b64',   data=default_glut_b64),
    ])

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    # ── Upload new image ───────────────────────────────────────────────────
    @app.callback(
        Output('img-original',     'figure',  allow_duplicate=True),
        Output('img-default',      'figure',  allow_duplicate=True),
        Output('img-edited',       'figure',  allow_duplicate=True),
        Output('store-original-b64', 'data',  allow_duplicate=True),
        Output('store-default-b64',  'data',  allow_duplicate=True),
        Output('store-edited-b64',   'data',  allow_duplicate=True),
        Output('store-constraint', 'data',    allow_duplicate=True),
        Output('store-img-b64-js', 'data',    allow_duplicate=True),
        Output('status-box',       'children', allow_duplicate=True),
        Input('upload-image',      'contents'),
        prevent_initial_call=True
    )
    def on_upload(contents):
        if contents is None:
            raise PreventUpdate
        new_img  = b64_to_np(contents)
        _img[0]  = new_img          # swap shared image reference
        editor.reset()
        hw           = new_img.shape[:2]
        orig_b64     = np_to_b64(new_img)
        default_b64  = np_to_b64(editor.apply_to_image(new_img))
        return (
            _image_figure(orig_b64,    'original', img_hw=hw),
            _image_figure(default_b64, 'default',  img_hw=hw),
            _image_figure(default_b64, 'edited',   img_hw=hw),
            orig_b64,
            default_b64,
            default_b64,
            {'x_in': None},
            orig_b64,                       # refresh the magnifier's source image
            [html.Span('✓ New image loaded. Editor reset.',
                       style={'color': AMBER})],
        )

    # ── Click image OR manual RGB entry → set x_e ────────────────────────
    @app.callback(
        Output('store-constraint', 'data'),
        Output('swatch-in',        'style'),
        Output('label-in',         'children'),
        Output('input-r',          'value'),
        Output('input-g',          'value'),
        Output('input-b',          'value'),
        Input('img-original',      'clickData'),
        Input('btn-set-xe',        'n_clicks'),
        State('store-constraint',  'data'),
        State('input-r',           'value'),
        State('input-g',           'value'),
        State('input-b',           'value'),
        prevent_initial_call=True
    )
    def pick_input_color(click_data, _btn, constraint, r_val, g_val, b_val):
        triggered = ctx.triggered_id

        if triggered == 'img-original':
            if click_data is None:
                return no_update, no_update, no_update, no_update, no_update, no_update

            pt = click_data['points'][0]
            px = int(round(pt.get('x', 0)))
            py = int(round(pt.get('y', 0)))

            # Plotly y: 0=bottom → h=top; numpy y: 0=top → flip
            h, w, _ = _img[0].shape
            px    = int(np.clip(px,         0, w - 1))
            py_np = int(np.clip(h - 1 - py, 0, h - 1))

            rgb_255 = _img[0][py_np, px] * 255
            rgb_01  = _img[0][py_np, px].tolist()
            r8, g8, b8 = int(rgb_255[0]), int(rgb_255[1]), int(rgb_255[2])

        else:  # btn-set-xe: read from number inputs
            try:
                r8 = int(np.clip(int(r_val), 0, 255)) if r_val is not None else 0
                g8 = int(np.clip(int(g_val), 0, 255)) if g_val is not None else 0
                b8 = int(np.clip(int(b_val), 0, 255)) if b_val is not None else 0
            except (TypeError, ValueError):
                return no_update, no_update, no_update, no_update, no_update, no_update
            rgb_01 = [r8 / 255.0, g8 / 255.0, b8 / 255.0]

        hex_col = f'#{r8:02x}{g8:02x}{b8:02x}'
        constraint['x_in'] = rgb_01
        label = f'{r8}, {g8}, {b8}'
        return constraint, swatch_style(hex_col), label, r8, g8, b8

    # ── Colour picker label ────────────────────────────────────────────────
    @app.callback(
        Output('label-out', 'children'),
        Input('picker-out', 'value'),
        prevent_initial_call=True
    )
    def update_out_label(hex_val):
        r, g, b = int(hex_val[1:3],16), int(hex_val[3:5],16), int(hex_val[5:7],16)
        return f'{r}, {g}, {b}'

    # ── Slider labels ──────────────────────────────────────────────────────
    @app.callback(
        Output('label-K',   'children'),
        Output('label-str', 'children'),
        Input('slider-K',   'value'),
        Input('slider-str', 'value'),
    )
    def update_slider_labels(k, s):
        return f'K GAUSSIANS  ·  {k}', f'STRENGTH  ·  {s:.2f}'

    # ── Apply / Reset ──────────────────────────────────────────────────────
    @app.callback(
        Output('img-edited',       'figure'),
        Output('cube-graph',       'figure'),
        Output('status-box',       'children'),
        Output('store-constraint', 'data',    allow_duplicate=True),
        Output('store-edited-b64', 'data',    allow_duplicate=True),
        Input('btn-apply',         'n_clicks'),
        Input('btn-reset',         'n_clicks'),
        State('store-constraint',  'data'),
        State('picker-out',        'value'),
        State('slider-K',          'value'),
        State('slider-str',        'value'),
        State('store-default-b64', 'data'),
        prevent_initial_call=True
    )
    def apply_or_reset(n_apply, n_reset, constraint, hex_out, K, strength,
                       cur_default_b64):
        triggered = ctx.triggered_id
        hw = _img[0].shape[:2]

        if triggered == 'btn-reset':
            editor.reset()
            status = [html.Span('↺ Parameters restored to checkpoint.',
                                style={'color': AMBER})]
            return (
                _image_figure(cur_default_b64, 'edited', img_hw=hw),
                build_cube_figure(editor),
                status,
                {'x_in': None},
                cur_default_b64,
            )

        x_in = constraint.get('x_in')
        if x_in is None:
            status = [html.Span('⚠ Click the original image to select c_in first.',
                                style={'color': AMBER})]
            return no_update, no_update, status, no_update, no_update

        r = int(hex_out[1:3], 16) / 255.0
        g = int(hex_out[3:5], 16) / 255.0
        b = int(hex_out[5:7], 16) / 255.0
        x_out = [r, g, b]

        info       = editor.edit_single(x_in, x_out, K=K, strength=strength)
        edited     = editor.apply_to_image(_img[0])
        edited_b64 = np_to_b64(edited)

        if info['status'] == 'already satisfied':
            status_text  = '✓ Constraint already satisfied — no update needed.'
            status_color = GREEN
        else:
            eb = info['error_before']
            ea = info['error_after']
            status_text  = (f"✓ Edit applied  ·  Gaussians {info['topk_indices']}\n"
                            f"  Δ before: {eb:.2f}  →  after: {ea:.2f}  (Δ255)")
            status_color = GREEN if ea < eb else AMBER

        status = [html.Pre(status_text,
                           style={'color': status_color, 'margin': 0,
                                  'fontSize': '11px', 'whiteSpace': 'pre-wrap'})]
        return (
            _image_figure(edited_b64, 'edited', img_hw=hw),
            build_cube_figure(editor, [(x_in, x_out)]),
            status,
            constraint,
            edited_b64,
        )

    # ── Save model ─────────────────────────────────────────────────────────
    @app.callback(
        Output('status-box', 'children', allow_duplicate=True),
        Input('btn-save', 'n_clicks'),
        prevent_initial_call=True
    )
    def save_model(n_clicks):
        out_path = model_path.replace('.pth', '_edited.pth')
        torch.save({'model_state_dict': editor.model.state_dict()}, out_path)
        return [html.Span(f'⬇ Saved → {os.path.basename(out_path)}',
                          style={'color': GREEN})]

    # ── Download all three images as ZIP ───────────────────────────────────
    @app.callback(
        Output('download-zip',       'data'),
        Input('btn-download-all',    'n_clicks'),
        State('store-original-b64',  'data'),
        State('store-default-b64',   'data'),
        State('store-edited-b64',    'data'),
        prevent_initial_call=True
    )
    def download_all(n_clicks, orig_b64, default_b64, edited_b64):
        def strip(b64):
            return base64.b64decode(b64.split(',', 1)[1])

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('original.png',     strip(orig_b64))
            zf.writestr('default_glut.png', strip(default_b64))
            zf.writestr('edited_glut.png',  strip(edited_b64))
        buf.seek(0)
        return dcc.send_bytes(buf.read(), filename='glut_images.zip')

    # ── Model selection ────────────────────────────────────────────────────
    @app.callback(
        Output('img-default',        'figure',  allow_duplicate=True),
        Output('img-edited',         'figure',  allow_duplicate=True),
        Output('cube-graph',         'figure',  allow_duplicate=True),
        Output('store-default-b64',  'data',    allow_duplicate=True),
        Output('store-edited-b64',   'data',    allow_duplicate=True),
        Output('store-constraint',   'data',    allow_duplicate=True),
        Output('status-box',         'children', allow_duplicate=True),
        Input('dropdown-model',      'value'),
        prevent_initial_call=True
    )
    def on_model_select(model_name):
        if model_name is None:
            raise PreventUpdate
        new_model_path = os.path.join(models_dir, model_name)
        checkpoint = torch.load(new_model_path, map_location='cpu')
        editor.model.load_state_dict(checkpoint['model_state_dict'])
        editor._snapshot_params()
        hw = _img[0].shape[:2]
        default_b64 = np_to_b64(editor.apply_to_image(_img[0]))
        return (
            _image_figure(default_b64, 'default', img_hw=hw),
            _image_figure(default_b64, 'edited',  img_hw=hw),
            build_cube_figure(editor),
            default_b64,
            default_b64,
            {'x_in': None},
            [html.Span(f'✓ Loaded model: {model_name}', style={'color': AMBER})],
        )

    # ── Magnifier loupe over the ORIGINAL image (clientside) ───────────────
    # Draws a zoomed view of the pixel under the cursor so colours can be
    # picked precisely; the actual pick is still the image click handled above.
    app.clientside_callback(
        """
        function(imgB64) {
            const graphDiv = document.getElementById('img-original');
            if (!graphDiv) return window.dash_clientside.no_update;

            const magnifier = document.getElementById('magnifier');
            const ZOOM  = 12;
            const MAG_R = 40;   // magnifier radius, px

            // Preload the image into an off-screen canvas so we can read pixels.
            let canvas = document.createElement('canvas');
            let ctxC   = canvas.getContext('2d');
            let img    = new Image();
            img.src    = imgB64;
            img.onload = function() {
                canvas.width  = img.naturalWidth;
                canvas.height = img.naturalHeight;
                ctxC.drawImage(img, 0, 0);
            };

            graphDiv.addEventListener('mousemove', function(e) {
                const plotArea = graphDiv.querySelector('.nsewdrag');
                if (!plotArea) return;
                const rect = plotArea.getBoundingClientRect();

                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                if (mx < 0 || my < 0 || mx > rect.width || my > rect.height) {
                    magnifier.style.display = 'none';
                    return;
                }

                // Ratio between the real image size and the displayed size.
                const scaleX = canvas.width  / rect.width;
                const scaleY = canvas.height / rect.height;
                const imgX = Math.floor(mx * scaleX);
                const imgY = Math.floor(my * scaleY);

                // Copy a small region around the cursor into the loupe canvas.
                const size = MAG_R * 2;
                let magCanvas = magnifier.querySelector('canvas');
                if (!magCanvas) {
                    magCanvas = document.createElement('canvas');
                    magCanvas.style.cssText = 'border-radius:50%;display:block';
                    magnifier.appendChild(magCanvas);
                }
                const mc = magCanvas.getContext('2d');

                const srcSize    = Math.floor(size / ZOOM) | 1;   // force odd
                const halfSrc    = Math.floor(srcSize / 2);
                const actualSize = srcSize * ZOOM;                // exact integer scale

                magCanvas.width        = actualSize;
                magCanvas.height       = actualSize;
                magCanvas.style.width  = actualSize + 'px';
                magCanvas.style.height = actualSize + 'px';

                mc.imageSmoothingEnabled = false;
                mc.drawImage(canvas, imgX - halfSrc, imgY - halfSrc,
                             srcSize, srcSize, 0, 0, actualSize, actualSize);

                // Crosshair box on the centre pixel.
                const boxLeft = halfSrc * ZOOM + 0.5;
                const boxTop  = halfSrc * ZOOM + 0.5;
                mc.strokeStyle = 'rgba(255, 255, 255, 0.9)';
                mc.lineWidth = 1.5;
                mc.strokeRect(boxLeft, boxTop, ZOOM, ZOOM);
                mc.strokeStyle = 'rgba(0, 0, 0, 0.5)';
                mc.lineWidth = 0.5;
                mc.strokeRect(boxLeft - 1, boxTop - 1, ZOOM + 2, ZOOM + 2);

                // Follow the cursor, positioned relative to the panel.
                const parentRect = graphDiv.parentElement.getBoundingClientRect();
                const px = e.clientX - parentRect.left;
                const py = e.clientY - parentRect.top;
                const offset = MAG_R + 12;
                magnifier.style.left    = (px + offset) + 'px';
                magnifier.style.top     = (py - MAG_R)  + 'px';
                magnifier.style.display = 'block';
            });

            graphDiv.addEventListener('mouseleave', function() {
                magnifier.style.display = 'none';
            });

            return window.dash_clientside.no_update;
        }
        """,
        Output('store-img-b64-js', 'data'),
        Input('store-img-b64-js',  'data'),
    )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GLUT Interactive Editor')
    parser.add_argument('--pretrained_model', type=str, default='./pretrained_models/glut/7luts/GLUT_LUT01_After_Effects_LUTs_Contrast_32_psnr52.64_dE0.3678.pth')
    parser.add_argument('--image',            type=str, default=None,
                        help='Optional starting image (can also upload in GUI)')
    parser.add_argument('--num_gaussians',    type=int, default=32)
    parser.add_argument('--residual',         type=bool, default=True)
    parser.add_argument('--port',             type=int, default=8050)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from train_glut import GLUT3D
    model = GLUT3D(
        num_gaussians=args.num_gaussians,
        residual=args.residual,
        logger=None
    ).to(device)
    checkpoint = torch.load(args.pretrained_model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if args.image:
        img_np = load_image(args.image)
    else:
        # Blank grey placeholder until the user uploads
        img_np = np.full((256, 256, 3), 0.2, dtype=np.float32)

    model_path = args.pretrained_model
    models_dir = os.path.dirname(model_path) or 'models'
    editor     = GLUTEditor(model, device)

    print(f"\n  GLUT Editor running →  http://127.0.0.1:{args.port}\n")
    app = build_app(editor, img_np, model_path,
                    models_dir=models_dir,
                    num_gaussians=args.num_gaussians,
                    residual=args.residual)
    app.run(debug=False, port=args.port)
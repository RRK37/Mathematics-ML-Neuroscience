import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# --- Configuration ---
# Signal s(t) — change this to experiment
t = np.linspace(-2, 2, 2000)
s = np.sin(2 * np.pi * t) + 0.5 * np.sin(2 * np.pi * 7 * t)

f_min, f_max, f_step = 0.1, 15.0, 0.1
freqs_slider = np.arange(f_min, f_max + f_step / 2, f_step)

# --- Precompute |S(f)| for slider colorbar ---
spectrum_mag = np.array([
    abs(np.trapezoid(s * np.exp(-2j * np.pi * fr * t), t))
    for fr in freqs_slider
])
mag_max = spectrum_mag.max() if spectrum_mag.max() > 0 else 1.0


def mag_to_color(mag):
    """Map magnitude to a color from dark charcoal (0) to vivid amber (max)."""
    norm = mag / mag_max
    r = int(42 + norm * (245 - 42))
    g = int(42 + norm * (166 - 42))
    b = int(48 + norm * (35 - 48))
    return f'rgb({r},{g},{b})'


def build_gradient_css():
    """Build a CSS linear-gradient encoding |S(f)| across the slider track."""
    stops = []
    for i, fr in enumerate(freqs_slider):
        pct = (fr - f_min) / (f_max - f_min) * 100
        color = mag_to_color(spectrum_mag[i])
        stops.append(f'{color} {pct:.1f}%')
    return f'linear-gradient(90deg, {", ".join(stops)})'


gradient = build_gradient_css()

app = Dash(__name__)

app.layout = html.Div([
    # Google Fonts
    html.Link(
        rel='stylesheet',
        href='https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,500;0,8..60,700;1,8..60,400&family=JetBrains+Mono:wght@400;600&display=swap'
    ),

    # Header
    html.Div([
        html.H1('Fourier Transform', style={
            'margin': '0', 'fontSize': '2rem', 'fontWeight': '700',
            'fontFamily': '"Source Serif 4", Georgia, serif',
            'color': '#1a1a1a', 'letterSpacing': '-0.02em',
        }),
        html.P(
            'Visualising the integrand  S(f) = ∫ s(t) · e⁻²ʲᵖᶠᵗ dt',
            style={
                'margin': '4px 0 0', 'fontSize': '1rem',
                'fontFamily': '"JetBrains Mono", monospace',
                'color': '#666', 'fontWeight': '400',
            }
        ),
    ], style={'padding': '28px 48px 0'}),

    # Result readout
    html.Div(id='result-readout', style={
        'padding': '12px 48px 0',
        'fontFamily': '"JetBrains Mono", monospace',
        'fontSize': '0.95rem', 'color': '#1a1a1a',
    }),

    # Slider section
    html.Div([
        html.Div([
            html.Span('f', style={
                'fontFamily': '"Source Serif 4", Georgia, serif',
                'fontStyle': 'italic', 'fontSize': '1.15rem',
                'fontWeight': '500', 'color': '#1a1a1a',
            }),
            html.Span(' (Hz)', style={
                'fontFamily': '"JetBrains Mono", monospace',
                'fontSize': '0.8rem', 'color': '#888', 'marginLeft': '2px',
            }),
        ], style={'marginBottom': '6px'}),
        dcc.Slider(
            id='freq-slider',
            min=f_min, max=f_max, step=f_step,
            value=3.0,
            marks={i: {'label': str(i), 'style': {
                'fontFamily': '"JetBrains Mono", monospace',
                'fontSize': '0.75rem', 'color': '#999',
            }} for i in range(1, int(f_max) + 1, 2)},
            tooltip={'always_visible': False},
            className='spectrum-slider',
        ),
        html.Div([
            html.Span('low |S(f)|', style={'color': '#888', 'fontSize': '0.7rem'}),
            html.Span('high |S(f)|', style={'color': '#f5a623', 'fontSize': '0.7rem'}),
        ], style={
            'display': 'flex', 'justifyContent': 'space-between',
            'fontFamily': '"JetBrains Mono", monospace',
            'padding': '2px 8px 0',
        }),
    ], style={'padding': '16px 48px 0'}),

    # Plot
    dcc.Graph(id='plot', style={'height': '70vh'},
              config={'displayModeBar': True, 'displaylogo': False}),

], style={
    'background': '#fafaf8',
    'minHeight': '100vh',
})

# Custom CSS for the slider track gradient
app.index_string = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Fourier Transform</title>
    {%favicon%}
    {%css%}
    <style>
        body { margin: 0; background: #fafaf8; }
        .spectrum-slider .rc-slider-track {
            background: transparent !important;
        }
        .spectrum-slider .rc-slider-rail {
            background: ''' + gradient + ''' !important;
            height: 10px !important;
            border-radius: 5px !important;
            opacity: 0.85 !important;
        }
        .spectrum-slider .rc-slider-handle {
            width: 20px !important;
            height: 20px !important;
            margin-top: -6px !important;
            border: 3px solid #1a1a1a !important;
            background: #fafaf8 !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.2) !important;
            opacity: 1 !important;
        }
        .spectrum-slider .rc-slider-handle:hover,
        .spectrum-slider .rc-slider-handle:active,
        .spectrum-slider .rc-slider-handle-dragging {
            border-color: #f5a623 !important;
            box-shadow: 0 0 0 3px rgba(245,166,35,0.2) !important;
        }
        .spectrum-slider .rc-slider-step {
            height: 10px !important;
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>'''


@app.callback(
    Output('plot', 'figure'),
    Output('result-readout', 'children'),
    Input('freq-slider', 'value'),
)
def update(f):
    kernel = np.exp(-2j * np.pi * f * t)
    product = s * kernel
    S_f = np.trapezoid(product, t)

    half = 0.55 / f
    t_lo, t_hi = -half, half

    mono = '"JetBrains Mono", monospace'
    serif = '"Source Serif 4", Georgia, serif'

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=s, name='  s(t)',
        line=dict(color='#2b5b8a', width=2.5),
        hovertemplate='t = %{x:.4f}<br>s(t) = %{y:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=t, y=kernel.real, name='  Re{ e⁻²ʲᵖᶠᵗ }',
        line=dict(color='#1a9a5a', width=1.8),
        hovertemplate='t = %{x:.4f}<br>cos(-2πft) = %{y:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=t, y=kernel.imag, name='  Im{ e⁻²ʲᵖᶠᵗ }',
        line=dict(color='#1a9a5a', width=1.4, dash='dot'), opacity=0.55,
        hovertemplate='t = %{x:.4f}<br>sin(-2πft) = %{y:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=t, y=product.real, name='  Re{ s·e⁻²ʲᵖᶠᵗ }',
        line=dict(color='#c0392b', width=2),
        fill='tozeroy', fillcolor='rgba(192,57,43,0.08)',
        hovertemplate='t = %{x:.4f}<br>Re = %{y:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=t, y=product.imag, name='  Im{ s·e⁻²ʲᵖᶠᵗ }',
        line=dict(color='#d4770a', width=1.4, dash='dot'), opacity=0.55,
        fill='tozeroy', fillcolor='rgba(212,119,10,0.05)',
        hovertemplate='t = %{x:.4f}<br>Im = %{y:.4f}<extra></extra>',
    ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text='t  (seconds)', font=dict(family=mono, size=13, color='#666')),
            range=[t_lo, t_hi],
            gridcolor='rgba(0,0,0,0.06)', gridwidth=1,
            zeroline=True, zerolinecolor='rgba(0,0,0,0.15)', zerolinewidth=1.5,
            tickfont=dict(family=mono, size=11, color='#999'),
        ),
        yaxis=dict(
            title=dict(text='Amplitude', font=dict(family=mono, size=13, color='#666')),
            gridcolor='rgba(0,0,0,0.06)', gridwidth=1,
            zeroline=True, zerolinecolor='rgba(0,0,0,0.15)', zerolinewidth=1.5,
            tickfont=dict(family=mono, size=11, color='#999'),
        ),
        plot_bgcolor='#fafaf8',
        paper_bgcolor='#fafaf8',
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
            font=dict(family=mono, size=12, color='#1a1a1a'),
            bgcolor='rgba(250,250,248,0.9)',
            bordercolor='rgba(0,0,0,0.08)', borderwidth=1,
        ),
        margin=dict(t=20, b=50, l=60, r=30),
        hoverlabel=dict(
            bgcolor='#1a1a1a', font_color='#fafaf8',
            font=dict(family=mono, size=12),
            bordercolor='#1a1a1a',
        ),
    )

    # Result readout
    readout = html.Div([
        html.Span('S', style={
            'fontFamily': serif, 'fontStyle': 'italic',
            'fontWeight': '500', 'fontSize': '1.05rem',
        }),
        html.Span(f'({f:.1f})', style={'fontFamily': mono}),
        html.Span('  =  ', style={'color': '#999'}),
        html.Span(f'{S_f.real:+.4f}', style={'fontWeight': '600'}),
        html.Span(f' {S_f.imag:+.4f}', style={'fontWeight': '600'}),
        html.Span('j', style={
            'fontFamily': serif, 'fontStyle': 'italic', 'fontWeight': '500',
        }),
        html.Span('    |    ', style={'color': '#ccc'}),
        html.Span('|', style={'color': '#999'}),
        html.Span('S', style={
            'fontFamily': serif, 'fontStyle': 'italic', 'fontWeight': '500',
        }),
        html.Span('|', style={'color': '#999'}),
        html.Span(f' = {abs(S_f):.4f}', style={'fontWeight': '600'}),
        html.Span('    |    ', style={'color': '#ccc'}),
        html.Span('∠ ', style={'color': '#999'}),
        html.Span(f'{np.degrees(np.angle(S_f)):.1f}°', style={'fontWeight': '600'}),
    ])

    return fig, readout


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)

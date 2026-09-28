import torch
import argparse
import os
import numpy as np
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

from scipy.interpolate import griddata
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import setup_logging
from train_glut import GLUT3D
from data.apply_cube import CubeFileTransform

class GaussianLUT3DVisualizer:
    """
    3D Gaussian LUT visualization tool
    visualize the prediction error inside the RGB cube
    """
    
    def __init__(self, 
                 model, 
                 save_dir='./results/vis',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        self.save_dir = save_dir
        
    def generate_test_points(self, num_points=10000, strategy='uniform'):
        """
        Generate test points
        
        Args:
            num_points: number of points
            strategy: sampling strategy
                - 'uniform': uniform sampling
                - 'grid': grid sampling
                - 'random': random sampling
                - 'stratified': stratified sampling
        """
        if strategy == 'uniform':
            # uniform samples in [0, 255]
            points = np.random.uniform(0, 255, (num_points, 3))
            
        elif strategy == 'grid':
            # grid sampling
            n = int(np.ceil(num_points ** (1/3)))
            x = np.linspace(0, 255, n)
            y = np.linspace(0, 255, n)
            z = np.linspace(0, 255, n)
            X, Y, Z = np.meshgrid(x, y, z)
            points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
            
        elif strategy == 'stratified':
            # stratified sampling (covers the whole cube)
            n_per_dim = int(np.ceil(num_points ** (1/3)))
            points = []
            for i in range(n_per_dim):
                for j in range(n_per_dim):
                    for k in range(n_per_dim):
                        # one random sample per sub-cube
                        p = np.array([
                            np.random.uniform(i*255/n_per_dim, (i+1)*255/n_per_dim),
                            np.random.uniform(j*255/n_per_dim, (j+1)*255/n_per_dim),
                            np.random.uniform(k*255/n_per_dim, (k+1)*255/n_per_dim)
                        ])
                        points.append(p)
            points = np.array(points[:num_points])
            
        else:  # random
            points = np.random.rand(num_points, 3) * 255
            
        return points / 255.0  # normalize to [0, 1]
    
    def compute_errors(self, test_points, target_transform=None):
        """
        Compute the prediction error
        
        Args:
            test_points: [N, 3] test points (in [0, 1])
            target_transform: target transform function; identity if None
            
        Returns:
            errors_dict: dict of error metrics
        """
        with torch.no_grad():
            # to tensor
            points_tensor = torch.FloatTensor(test_points).to(self.device)
            
            # model prediction
            if hasattr(self.model, 'forward'):
                pred_rgb = self.model(points_tensor)
            else:
                pred_rgb = self.model(points_tensor)
            
            # compute the target
            if target_transform is None:
                target_rgb = points_tensor  # identity
            else:
                target_rgb = target_transform(points_tensor)
            
            # to numpy
            pred_rgb = pred_rgb.cpu().numpy()
            target_rgb = target_rgb.cpu().numpy() if torch.is_tensor(target_rgb) else target_rgb
            
            # compute the error metrics
            # 1. Euclidean distance
            euclidean_dist = np.linalg.norm(pred_rgb - target_rgb, axis=1)
            
            # 2. per-channel error
            channel_errors = np.abs(pred_rgb - target_rgb)
            
            # 3. relative error
            rel_error = channel_errors / (target_rgb + 1e-8)
            
            # 4. angular error (colour-direction deviation)
            pred_norm = pred_rgb / (np.linalg.norm(pred_rgb, axis=1, keepdims=True) + 1e-8)
            target_norm = target_rgb / (np.linalg.norm(target_rgb, axis=1, keepdims=True) + 1e-8)
            dot_product = np.sum(pred_norm * target_norm, axis=1)
            angular_error = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi
            
            return {
                'points': test_points * 255,  # back to 0-255 for plotting
                'pred_rgb': pred_rgb * 255,
                'target_rgb': target_rgb * 255,
                'euclidean_dist': euclidean_dist,
                'channel_errors': channel_errors * 255,
                'rel_error': rel_error,
                'angular_error': angular_error,
                'mse': np.mean(euclidean_dist ** 2),
                'mae': np.mean(euclidean_dist),
                'max_error': np.max(euclidean_dist)
            }
        
    def compute_errors_with_cube(self, test_points, cube_file_path, interpolation='trilinear'):
        """
        Prediction error using a .cube file as ground truth
        
        Args:
            test_points: [N, 3] test points (in [0, 1])
            cube_file_path: path to the .cube file
            interpolation: interpolation ('trilinear' | 'nearest')
            
        Returns:
            errors_dict: dict of error metrics
        """
        # build the .cube transform
        cube_transform = CubeFileTransform(cube_file_path, interpolation)
        
        with torch.no_grad():
            # to tensor
            points_tensor = torch.FloatTensor(test_points).to(self.device)
            
            # model prediction
            if hasattr(self.model, 'forward'):
                pred_rgb = self.model(points_tensor)
            else:
                pred_rgb = self.model(points_tensor)
            
            # target from the .cube transform
            target_rgb = cube_transform(points_tensor)
            
            # to numpy
            pred_rgb_np = pred_rgb.cpu().numpy()
            target_rgb_np = target_rgb.cpu().numpy()

            # from metrics import calc_metrics
            # metrics = calc_metrics(pred_rgb, target_rgb)
            # metrics_org = calc_metrics(pred_rgb, points_tensor)
            
            # compute the error metrics
            # 1. Euclidean distance
            euclidean_dist = np.linalg.norm(pred_rgb_np - target_rgb_np, axis=1)
            
            # 2. per-channel error
            channel_errors = np.abs(pred_rgb_np - target_rgb_np)
            
            # 3. relative error
            rel_error = channel_errors / (target_rgb_np + 1e-8)
            
            # 4. angular error (colour-direction deviation)
            pred_norm = pred_rgb_np / (np.linalg.norm(pred_rgb_np, axis=1, keepdims=True) + 1e-8)
            target_norm = target_rgb_np / (np.linalg.norm(target_rgb_np, axis=1, keepdims=True) + 1e-8)
            dot_product = np.sum(pred_norm * target_norm, axis=1)
            angular_error = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi
            
            # 5. Delta E 2000 (needs an extra library)
            try:
                from utils.metrics import calc_deltaE
                delta_e = calc_deltaE(pred_rgb_np, target_rgb_np, color_space='sRGB', method='CIE 2000', CAT='CAT02', reduction='none')

                # from skimage.color import deltaE_ciede2000
                # import cv2
                
                # # convert to LAB for a more accurate colour difference
                # pred_lab = cv2.cvtColor((pred_rgb_np * 255).astype(np.uint8).reshape(-1,1,3), 
                #                         cv2.COLOR_RGB2LAB).reshape(-1, 3)
                # target_lab = cv2.cvtColor((target_rgb_np * 255).astype(np.uint8).reshape(-1,1,3), 
                #                         cv2.COLOR_RGB2LAB).reshape(-1, 3)
                
                # delta_e = []
                # for i in range(len(pred_lab)):
                #     de = deltaE_ciede2000(pred_lab[i], target_lab[i])
                #     delta_e.append(de)
                # delta_e = np.array(delta_e)
            except ImportError:
                print("WARNING: skimage/opencv not installed, skippingDelta E computation")
                delta_e = np.zeros_like(euclidean_dist)
            
            return {
                'points': test_points * 255,  # back to 0-255 for plotting
                'pred_rgb': pred_rgb_np * 255,
                'target_rgb': target_rgb_np * 255,
                'euclidean_dist': euclidean_dist,
                'channel_errors': channel_errors * 255,
                'rel_error': rel_error,
                'angular_error': angular_error,
                'delta_e': delta_e,
                'mse': np.mean(euclidean_dist ** 2),
                'mae': np.mean(euclidean_dist),
                'max_error': np.max(euclidean_dist),
                'cube_file': cube_file_path,
                'cube_size': cube_transform.size
            }
    
    def visualize_3d_scatter(self, errors_dict, 
                            color_by='euclidean_dist',
                            title="3D Gaussian LUT Error Visualization",
                            save_path=None):
        """
        Interactive 3D scatter (plotly)
        
        Args:
            errors_dict: the dict returned by compute_errors
            color_by: how to colour the points
                - 'euclidean_dist': Euclidean distance
                - 'channel_r/g/b': a specific channel error
                - 'angular_error': angular error
                - 'gt': by ground-truth colour
                - 'pred': by predicted colour
        """
        points = errors_dict['points']
        
        # pick the colour
        if color_by == 'euclidean_dist':
            colors = errors_dict['euclidean_dist']
            colorbar_title = 'Euclidean Distance'
            colorscale = 'Viridis'
        elif color_by == 'angular_error':
            colors = errors_dict['angular_error']
            colorbar_title = 'Angular Error (degrees)'
            colorscale = 'viridis'
        elif color_by.startswith('channel_'):
            channel = color_by.split('_')[1]
            idx = {'r': 0, 'g': 1, 'b': 2}[channel]
            colors = errors_dict['channel_errors'][:, idx]
            colorbar_title = f'{channel.upper()} Channel Error'
            colorscale = 'Reds' if channel == 'r' else 'Greens' if channel == 'g' else 'Blues'
        elif color_by == 'gt':
            colors = errors_dict['target_rgb'] / 255
            colorbar_title = 'Ground Truth RGB'
            colorscale = 'Viridis'
        elif color_by == 'pred':
            colors = errors_dict['pred_rgb'] / 255
            colorbar_title = 'Predicted RGB'
            colorscale = 'Viridis'
        else:
            colors = errors_dict['euclidean_dist']
            colorbar_title = 'Error'
            colorscale = 'Viridis'
        
        # build the 3D scatter
        fig = go.Figure(data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=colors,
                    colorscale=colorscale,
                    colorbar=dict(title=colorbar_title),
                    showscale=True,
                    opacity=1.0
                ),
                text=[f'RGB: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})<br>'
                      f'Error: {e:.2f}<br>'
                      f'Pred: ({pred[0]:.1f}, {pred[1]:.1f}, {pred[2]:.1f})'
                      for p, e, pred in zip(points, errors_dict['euclidean_dist'], 
                                           errors_dict['pred_rgb'])],
                hoverinfo='text'
            )
        ])
        
        # add the RGB-cube edges
        cube_edges = [
            [[0,0,0], [255,0,0]],
            [[0,0,0], [0,255,0]],
            [[0,0,0], [0,0,255]],
            [[255,0,0], [255,255,0]],
            [[255,0,0], [255,0,255]],
            [[0,255,0], [255,255,0]],
            [[0,255,0], [0,255,255]],
            [[0,0,255], [255,0,255]],
            [[0,0,255], [0,255,255]],
            [[255,255,0], [255,255,255]],
            [[255,0,255], [255,255,255]],
            [[0,255,255], [255,255,255]]
        ]
        
        for edge in cube_edges:
            fig.add_trace(go.Scatter3d(
                x=[edge[0][0], edge[1][0]],
                y=[edge[0][1], edge[1][1]],
                z=[edge[0][2], edge[1][2]],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False,
                hoverinfo='none'
            ))
        
        # update the layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Red',
                yaxis_title='Green',
                zaxis_title='Blue',
                xaxis=dict(range=[0, 255]),
                yaxis=dict(range=[0, 255]),
                zaxis=dict(range=[0, 255]),
                aspectmode='cube'
            ),
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
        return fig
    
    def visualize_error_slices(self, errors_dict, n_slices=5, save_path=None):
        """
        Visualize slices through the RGB cube
        
        Args:
            errors_dict: the dict returned by compute_errors
            n_slices: number of slices per axis
        """
        points = errors_dict['points']
        errors = errors_dict['euclidean_dist']
        
        # create the subplots
        fig, axes = plt.subplots(n_slices, 3, figsize=(15, 4*n_slices))
        
        for i, channel in enumerate(['R', 'G', 'B']):
            for j in range(n_slices):
                slice_val = (j + 0.5) * 255 / n_slices
                
                if channel == 'R':
                    mask = (points[:, 0] > slice_val - 255/(2*n_slices)) & \
                           (points[:, 0] < slice_val + 255/(2*n_slices))
                    x_axis, y_axis = points[:, 1], points[:, 2]
                    xlabel, ylabel = 'Green', 'Blue'
                elif channel == 'G':
                    mask = (points[:, 1] > slice_val - 255/(2*n_slices)) & \
                           (points[:, 1] < slice_val + 255/(2*n_slices))
                    x_axis, y_axis = points[:, 0], points[:, 2]
                    xlabel, ylabel = 'Red', 'Blue'
                else:  # B
                    mask = (points[:, 2] > slice_val - 255/(2*n_slices)) & \
                           (points[:, 2] < slice_val + 255/(2*n_slices))
                    x_axis, y_axis = points[:, 0], points[:, 1]
                    xlabel, ylabel = 'Red', 'Green'
                
                if mask.sum() > 0:
                    scatter = axes[j, i].scatter(
                        x_axis[mask], y_axis[mask], 
                        c=errors[mask], 
                        cmap='viridis',
                        s=20,
                        alpha=1.0,
                        vmin=0,
                        vmax=np.percentile(errors, 95)
                    )
                    axes[j, i].set_title(f'{channel}={slice_val:.1f}')
                    axes[j, i].set_xlabel(xlabel)
                    axes[j, i].set_ylabel(ylabel)
                    axes[j, i].set_xlim(0, 255)
                    axes[j, i].set_ylim(0, 255)
                    
                    if j == 0 and i == 2:
                        plt.colorbar(scatter, ax=axes[j, i], label='Error')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_error_histograms(self, errors_dict, save_path=None):
        """
        Error-distribution histograms
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Euclidean distance
        axes[0, 0].hist(errors_dict['euclidean_dist'], bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Euclidean Distance')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Euclidean Distance Distribution\n'
                             f'Mean: {errors_dict["euclidean_dist"].mean():.2f}, '
                             f'Std: {errors_dict["euclidean_dist"].std():.2f}')
        
        # 2. per-channel error
        colors = ['red', 'green', 'blue']
        for i, (channel, color) in enumerate(zip(['R', 'G', 'B'], colors)):
            axes[0, 1].hist(errors_dict['channel_errors'][:, i], 
                           bins=50, alpha=0.5, color=color, label=channel)
        axes[0, 1].set_xlabel('Channel Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Per-Channel Error Distribution')
        axes[0, 1].legend()
        
        # 3. angular error
        axes[0, 2].hist(errors_dict['angular_error'], bins=50, alpha=0.7, color='purple')
        axes[0, 2].set_xlabel('Angular Error (degrees)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title(f'Angular Error Distribution\n'
                             f'Mean: {errors_dict["angular_error"].mean():.2f}°')
        
        # 4. relative error
        for i, (channel, color) in enumerate(zip(['R', 'G', 'B'], colors)):
            axes[1, 0].hist(errors_dict['rel_error'][:, i], 
                           bins=50, alpha=0.5, color=color, label=channel, range=(0, 1))
        axes[1, 0].set_xlabel('Relative Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Relative Error Distribution')
        axes[1, 0].legend()
        
        # 5. cumulative error
        sorted_errors = np.sort(errors_dict['euclidean_dist'])
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 1].plot(sorted_errors, cumulative)
        axes[1, 1].set_xlabel('Euclidean Distance')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. error vs. luminance
        brightness = np.mean(errors_dict['points'], axis=1)
        axes[1, 2].scatter(brightness, errors_dict['euclidean_dist'], 
                          alpha=0.3, s=1, c='black')
        axes[1, 2].set_xlabel('Brightness')
        axes[1, 2].set_ylabel('Error')
        axes[1, 2].set_title('Error vs Brightness')
        
        # add a trend line
        z = np.polyfit(brightness, errors_dict['euclidean_dist'], 1)
        p = np.poly1d(z)
        axes[1, 2].plot(np.sort(brightness), p(np.sort(brightness)), 
                       'r-', linewidth=2, label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
        axes[1, 2].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_gaussian_parameters(self, save_path=None):
        """
        Gaussian-parameter distributions
        """
        # read the Gaussian parameters
        if hasattr(self.model, 'get_gaussian_info'):
            info = self.model.get_gaussian_info()
            positions = info['positions'] * 255  # back to 0-255
            stds = info['stds'] * 255 if 'stds' in info else None
            opacities = info['opacities'] if 'opacities' in info else None
        else:
            positions = self.model.positions.detach().cpu().numpy() * 255
            stds = None
            opacities = self.model.opacity_activation(
                self.model.opacities_logit
            ).detach().cpu().numpy() if hasattr(self.model, 'opacities_logit') else None
        
        fig = plt.figure(figsize=(20, 10))
        
        # 1. Gaussian centres
        ax1 = fig.add_subplot(231, projection='3d')
        if opacities is not None:
            scatter = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                 c=opacities.flatten(), cmap='viridis', s=50, alpha=0.6)
            plt.colorbar(scatter, ax=ax1, label='Opacity')
        else:
            ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                       c='blue', s=50, alpha=0.6)
        ax1.set_xlabel('Red')
        ax1.set_ylabel('Green')
        ax1.set_zlabel('Blue')
        ax1.set_title('Gaussian Centers')
        ax1.set_xlim(0, 255)
        ax1.set_ylim(0, 255)
        ax1.set_zlim(0, 255)
        
        # 2. opacity distribution
        if opacities is not None:
            ax2 = fig.add_subplot(232)
            ax2.hist(opacities.flatten(), bins=50, alpha=0.7, color='orange')
            ax2.set_xlabel('Opacity')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Opacity Distribution')
        
        # 3. std-dev distribution
        if stds is not None:
            ax3 = fig.add_subplot(233)
            for i, (channel, color) in enumerate(zip(['R', 'G', 'B'], ['red', 'green', 'blue'])):
                ax3.hist(stds[:, i], bins=50, alpha=0.5, color=color, label=channel)
            ax3.set_xlabel('Standard Deviation')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Gaussian Std Dev Distribution')
            ax3.legend()
        
        # 4. Gaussian density (projection)
        ax4 = fig.add_subplot(234)
        from scipy.stats import gaussian_kde
        if len(positions) > 1:
            kde = gaussian_kde(positions[:, :2].T)
            x = np.linspace(0, 255, 100)
            y = np.linspace(0, 255, 100)
            X, Y = np.meshgrid(x, y)
            positions_grid = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions_grid).reshape(X.shape)
            ax4.imshow(Z, extent=[0, 255, 0, 255], origin='lower', cmap='viridis')
            ax4.set_xlabel('Red')
            ax4.set_ylabel('Green')
            ax4.set_title('Gaussian Density (RG projection)')
            plt.colorbar(ax4.images[0], ax=ax4, label='Density')
        
        # 5. Gaussian volume distribution
        if stds is not None:
            ax5 = fig.add_subplot(235)
            volumes = (4/3) * np.pi * np.prod(stds, axis=1)
            ax5.hist(np.log10(volumes + 1), bins=50, alpha=0.7, color='green')
            ax5.set_xlabel('log10(Volume + 1)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Gaussian Volume Distribution')
        
        # 6. colour-matrix-norm distribution
        if hasattr(self.model, 'color_matrices'):
            color_matrices = self.model.color_matrices.detach().cpu().numpy()
            matrix_norms = np.linalg.norm(color_matrices.reshape(-1, 9), axis=1)
            ax6 = fig.add_subplot(236)
            ax6.hist(matrix_norms, bins=50, alpha=0.7, color='purple')
            ax6.set_xlabel('Matrix Norm')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Color Matrix Norm Distribution')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_error_heatmap(self, errors_dict, resolution=32, save_path=None):
        """
        3D error heat map
        """
        points = errors_dict['points']
        errors = errors_dict['euclidean_dist']
        
        # build a 3D grid
        bins = np.linspace(0, 255, resolution + 1)
        grid_errors = np.zeros((resolution, resolution, resolution))
        grid_counts = np.zeros((resolution, resolution, resolution))
        
        # assign points to grid cells
        for point, error in zip(points, errors):
            i = np.digitize(point[0], bins[1:-1])
            j = np.digitize(point[1], bins[1:-1])
            k = np.digitize(point[2], bins[1:-1])
            
            if 0 <= i < resolution and 0 <= j < resolution and 0 <= k < resolution:
                grid_errors[i, j, k] += error
                grid_counts[i, j, k] += 1
        
        # mean error
        mask = grid_counts > 0
        grid_errors[mask] /= grid_counts[mask]
        
        # build the figure
        fig = plt.figure(figsize=(15, 5))
        
        # projections along each axis
        for idx, (axis, title) in enumerate(zip([0, 1, 2], ['R fixed', 'G fixed', 'B fixed'])):
            ax = fig.add_subplot(1, 3, idx + 1)
            
            # project along the chosen axis
            projection = np.mean(grid_errors, axis=axis)
            
            im = ax.imshow(projection.T, origin='lower', 
                          extent=[0, 255, 0, 255], 
                          cmap='viridis', aspect='auto')
            ax.set_xlabel(['G', 'R', 'R'][idx])
            ax.set_ylabel(['B', 'B', 'G'][idx])
            ax.set_title(f'Error Projection ({title})')
            plt.colorbar(im, ax=ax, label='Mean Error')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def comprehensive_visualization(self, cube_file_path, num_points=5000, save_prefix=None):
        """
        Full visualization report
        """
        # Generate test points
        test_points = self.generate_test_points(num_points, strategy='stratified')
        
        # compute the errors
        # errors_dict = self.compute_errors(test_points)

        errors_dict = self.compute_errors_with_cube(test_points, cube_file_path, interpolation='trilinear')
        
        print("=" * 80)
        print("Gaussian LUT error-analysis report")
        print("=" * 80)
        print(f"test points: {num_points}")
        print(f"MSE: {errors_dict['mse']:.4f}")
        print(f"MAE: {errors_dict['mae']:.4f}")
        print(f"Max Error: {errors_dict['max_error']:.4f}")
        print(f"Mean Angular Error: {errors_dict['angular_error'].mean():.2f}°")
        print()
        print("per-channel error:")
        for i, channel in enumerate(['R', 'G', 'B']):
            print(f"  {channel}: mean={errors_dict['channel_errors'][:, i].mean():.4f}, "
                  f"std={errors_dict['channel_errors'][:, i].std():.4f}")
        
        # 1. interactive 3D visualization
        print("\ninteractive 3D visualization...")
        fig1 = self.visualize_3d_scatter(
            errors_dict, 
            color_by='euclidean_dist',
            title="Gaussian LUT Error Distribution",
            save_path=os.path.join(self.save_dir, f"{save_prefix}_3d.html") if save_prefix else None
        )
        
        # 2. error slices
        print("error-slice plots...")
        self.visualize_error_slices(
            errors_dict, 
            n_slices=4,
            save_path=os.path.join(self.save_dir, f"{save_prefix}_slices.png") if save_prefix else None
        )
        
        # 3. error histogram
        print("error-distribution plots...")
        self.visualize_error_histograms(
            errors_dict,
            save_path=os.path.join(self.save_dir, f"{save_prefix}_histograms.png") if save_prefix else None
        )
        
        # 4. Gaussian-parameter plots
        print("Gaussian-parameter plots...")
        self.visualize_gaussian_parameters(
            save_path=os.path.join(self.save_dir, f"{save_prefix}_parameters.png") if save_prefix else None
        )
        
        # 5. error heat map
        print("error heat map...")
        self.create_error_heatmap(
            errors_dict,
            resolution=32,
            save_path=os.path.join(self.save_dir, f"{save_prefix}_heatmap.png") if save_prefix else None
        )
        
        return errors_dict



class EnhancedGaussianLUTVisualizer:
    """
    Extended Gaussian LUT visualizer
    draws each Gaussian's ellipse extent and opacity
    """
    
    def __init__(self, model, save_dir='./results/vis', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.model.eval()
        
    def get_gaussian_ellipsoids(self, num_samples=100):
        """
        Ellipsoid representation of every Gaussian
        
        Returns:
            ellipsoids: list of per-Gaussian ellipsoid dicts
        """
        # read the Gaussian parameters
        if hasattr(self.model, 'get_gaussian_info'):
            info = self.model.get_gaussian_info()
            positions = info['positions'] * 255
            if 'covariance' in info:
                covariances = info['covariance']
            else:
                # no covariance available: use a diagonal one
                stds = info.get('stds', np.ones((len(positions), 3)) * 0.1)
                covariances = np.array([np.diag(std**2) for std in stds])
            
            opacities = info.get('opacities', np.ones((len(positions), 1))).flatten()
            color_matrices = info.get('color_matrices', None)
            
        else:
            positions = self.model.positions.detach().cpu().numpy() * 255
            
            # covariance matrix
            if hasattr(self.model, 'get_covariance_matrix'):
                covariances = self.model.get_covariance_matrix().detach().cpu().numpy()
            else:
                # build a diagonal covariance from the scales
                if hasattr(self.model, 'scales'):
                    scales = self.model.scales.detach().cpu().numpy()
                elif hasattr(self.model, 'log_scales'):
                    scales = torch.exp(self.model.log_scales).detach().cpu().numpy()
                else:
                    scales = np.ones((len(positions), 3)) * 0.1
                covariances = np.array([np.diag(scale**2) for scale in scales])
            
            # opacity
            if hasattr(self.model, 'opacities_logit'):
                opacities = torch.sigmoid(self.model.opacities_logit).detach().cpu().numpy().flatten()
            else:
                opacities = np.ones(len(positions))
            
            # colour matrix
            if hasattr(self.model, 'color_matrices'):
                color_matrices = self.model.color_matrices.detach().cpu().numpy()
            else:
                color_matrices = None
        
        ellipsoids = []
        for i in range(len(positions)):
            # eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(covariances[i])
            
            # clamp eigenvalues to be positive
            eigvals = np.maximum(eigvals, 1e-6)
            
            # ellipsoid axes (2 std, ~95% of the mass)
            axes_lengths = 2 * np.sqrt(eigvals) * 255  # to the 0-255 range
            
            ellipsoids.append({
                'center': positions[i],
                'axes_lengths': axes_lengths,
                'rotation': eigvecs,
                'opacity': opacities[i],
                'covariance': covariances[i] * (255**2),  # scale the covariance to the 0-255 range
                'color_matrix': color_matrices[i] if color_matrices is not None else None,
                'eigenvalues': eigvals * (255**2)
            })
        
        return ellipsoids
    
    def plot_3d_ellipsoid_matplotlib(self, ax, ellipsoid, color='blue', alpha=1.0, 
                                     draw_axes=False, wireframe=True):
        """
        Draw an ellipsoid on a matplotlib 3D axis
        
        Args:
            ax: matplotlib 3D axis
            ellipsoid: ellipsoid info dict
            color: colour
            alpha: alpha
            draw_axes: whether to draw the principal axes
            wireframe: whether to use wireframe mode
        """
        center = ellipsoid['center']
        axes_lengths = ellipsoid['axes_lengths'] / 2  # radius
        rotation = ellipsoid['rotation']
        opacity = ellipsoid['opacity']
        
        # points on the unit sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        # grid form (suitable for plot_surface)
        u_grid, v_grid = np.meshgrid(u, v)
        x = np.cos(u_grid) * np.sin(v_grid)
        y = np.sin(u_grid) * np.sin(v_grid)
        z = np.cos(v_grid)
        
        # apply the scaling
        x_scaled = x * axes_lengths[0]
        y_scaled = y * axes_lengths[1]
        z_scaled = z * axes_lengths[2]
        
        # apply the rotation
        points = np.stack([x_scaled.flatten(), y_scaled.flatten(), z_scaled.flatten()])
        rotated_points = rotation @ points
        
        # reshape and translate to the centre
        X = rotated_points[0, :].reshape(x.shape) + center[0]
        Y = rotated_points[1, :].reshape(y.shape) + center[1]
        Z = rotated_points[2, :].reshape(z.shape) + center[2]
        
        # draw the ellipsoid surface
        if wireframe:
            ax.plot_wireframe(X, Y, Z, color=color, alpha=alpha * opacity, linewidth=0.5, rstride=2, cstride=2)
        else:
            ax.plot_surface(X, Y, Z, color=color, alpha=alpha * opacity, 
                          linewidth=0, antialiased=True, rstride=2, cstride=2)
        
        # draw the principal axes
        if draw_axes:
            for i in range(3):
                axis = rotation[:, i] * axes_lengths[i]
                ax.quiver(center[0], center[1], center[2],
                         axis[0], axis[1], axis[2],
                         color='red', linewidth=1, alpha=opacity,
                         arrow_length_ratio=0.1)
    
    def plot_3d_ellipsoid_points(self, ax, ellipsoid, color='blue', alpha=1.0, num_points=500):
        """
        Draw the ellipsoid as a point cloud (alternative)
        
        Args:
            ax: matplotlib 3D axis
            ellipsoid: ellipsoid info dict
            color: colour
            alpha: alpha
            num_points: number of points
        """
        center = ellipsoid['center']
        axes_lengths = ellipsoid['axes_lengths'] / 2
        rotation = ellipsoid['rotation']
        opacity = ellipsoid['opacity']
        
        # random points on the sphere
        phi = np.random.uniform(0, 2*np.pi, num_points)
        theta = np.random.uniform(0, np.pi, num_points)
        
        x = np.sin(theta) * np.cos(phi) * axes_lengths[0]
        y = np.sin(theta) * np.sin(phi) * axes_lengths[1]
        z = np.cos(theta) * axes_lengths[2]
        
        # apply the rotation
        points = np.stack([x, y, z])
        rotated_points = rotation @ points
        
        # translate to the centre
        X = rotated_points[0, :] + center[0]
        Y = rotated_points[1, :] + center[1]
        Z = rotated_points[2, :] + center[2]
        
        # draw the point cloud
        ax.scatter(X, Y, Z, c=color, alpha=alpha * opacity, s=1)
    
    def visualize_3d_matplotlib(self, ellipsoids, error_points=None, error_values=None,
                               title="3D Gaussian Distributions", save_path=None,
                               use_points=False):
        """
        3D visualization (matplotlib)
        
        Args:
            use_points: True -> point-cloud mode, False -> wireframe mode
        """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # plot the error points (if given)
        if error_points is not None and error_values is not None:
            scatter = ax.scatter(error_points[:, 0], error_points[:, 1], error_points[:, 2],
                               c=error_values, cmap='viridis', s=1, alpha=1.0,
                               vmin=0, vmax=np.percentile(error_values, 95) if len(error_values) > 0 else 50)
            plt.colorbar(scatter, ax=ax, label='Error', shrink=0.5)
        
        # draw every Gaussian ellipsoid
        for i, ellipsoid in enumerate(ellipsoids):
            # colour by opacity
            opacity = ellipsoid['opacity']
            
            # colour by the colour-matrix info if available
            if ellipsoid['color_matrix'] is not None:
                # colour by the matrix norm
                matrix_norm = np.linalg.norm(ellipsoid['color_matrix'])
                color = plt.cm.viridis(matrix_norm / 5)  # normalize
            else:
                # colour by opacity
                color = plt.cm.Blues(opacity * 0.8 + 0.2)
            
            if use_points:
                self.plot_3d_ellipsoid_points(ax, ellipsoid, color=color, alpha=1.0)
            else:
                self.plot_3d_ellipsoid_matplotlib(ax, ellipsoid, color=color, 
                                                 alpha=1.0, wireframe=True)
            
            # plot the centre point
            ax.scatter(ellipsoid['center'][0], ellipsoid['center'][1], ellipsoid['center'][2],
                      color='red', s=10 * opacity + 5, alpha=opacity)
        
        # axis labels / limits
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_title(title)
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)
        
        # equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig, ax
    
    def visualize_3d_plotly(self, ellipsoids, error_points=None, error_values=None,
                           title="3D Gaussian Distributions", save_path=None):
        """
        Interactive 3D visualization (plotly)
        """
        fig = go.Figure()
        
        # add the error points (if given)
        if error_points is not None and error_values is not None:
            fig.add_trace(go.Scatter3d(
                x=error_points[:, 0],
                y=error_points[:, 1],
                z=error_points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=error_values,
                    colorscale='viridis',
                    colorbar=dict(title='Error'),
                    showscale=True,
                    opacity=1.0
                ),
                name='Error Points',
                hoverinfo='none'
            ))
        
        # add every Gaussian ellipsoid
        for i, ellipsoid in enumerate(ellipsoids):
            opacity = ellipsoid['opacity']
            if opacity < 0.02:
                continue
            center = ellipsoid['center']
            axes = ellipsoid['axes_lengths'] / 2
            rotation = ellipsoid['rotation']
            
            # points on the ellipsoid
            u = np.linspace(0, 2*np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            u_grid, v_grid = np.meshgrid(u, v)
            
            x = axes[0] * np.cos(u_grid) * np.sin(v_grid)
            y = axes[1] * np.sin(u_grid) * np.sin(v_grid)
            z = axes[2] * np.cos(v_grid)
            
            # apply the rotation
            points = np.stack([x.flatten(), y.flatten(), z.flatten()])
            rotated_points = rotation @ points
            
            # reshape and translate
            X = rotated_points[0, :].reshape(x.shape) + center[0]
            Y = rotated_points[1, :].reshape(y.shape) + center[1]
            Z = rotated_points[2, :].reshape(z.shape) + center[2]
            
            # add the ellipsoid surface
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                opacity=opacity * 0.3,
                colorscale=[[0, f'rgba(100,100,100,{opacity})'], 
                           [1, f'rgba(255,100,100,{opacity})']],
                showscale=False,
                name=f'Gaussian {i}',
                hoverinfo='text',
                text=f'Gaussian {i}<br>'
                     f'Center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})<br>'
                     f'Axes: ({axes[0]*2:.1f}, {axes[1]*2:.1f}, {axes[2]*2:.1f})<br>'
                     f'Opacity: {opacity:.3f}'
            ))
            
            # add the centre point
            fig.add_trace(go.Scatter3d(
                x=[center[0]], y=[center[1]], z=[center[2]],
                mode='markers',
                marker=dict(size=5 * opacity + 3, color='red'),
                showlegend=False,
                hoverinfo='text',
                text=f'Center {i}'
            ))
        
        # add the RGB-cube edges
        cube_edges = [
            [[0,0,0], [255,0,0]], [[0,0,0], [0,255,0]], [[0,0,0], [0,0,255]],
            [[255,0,0], [255,255,0]], [[255,0,0], [255,0,255]],
            [[0,255,0], [255,255,0]], [[0,255,0], [0,255,255]],
            [[0,0,255], [255,0,255]], [[0,0,255], [0,255,255]],
            [[255,255,0], [255,255,255]], [[255,0,255], [255,255,255]],
            [[0,255,255], [255,255,255]]
        ]
        
        for edge in cube_edges:
            fig.add_trace(go.Scatter3d(
                x=[edge[0][0], edge[1][0]],
                y=[edge[0][1], edge[1][1]],
                z=[edge[0][2], edge[1][2]],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False,
                hoverinfo='none'
            ))
        
        # update the layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Red',
                yaxis_title='Green',
                zaxis_title='Blue',
                xaxis=dict(range=[0, 255]),
                yaxis=dict(range=[0, 255]),
                zaxis=dict(range=[0, 255]),
                aspectmode='cube'
            ),
            width=1000,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
        return fig
    

    def visualize_3d_plotly_o_error(self, error_points=None, error_values=None,
                           title="Nilut Error Distributions", save_path=None):
        """
        Interactive 3D visualization (plotly)
        """
        fig = go.Figure()
        
        # add the error points (if given)
        if error_points is not None and error_values is not None:
            fig.add_trace(go.Scatter3d(
                x=error_points[:, 0],
                y=error_points[:, 1],
                z=error_points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=error_values,
                    colorscale='viridis',
                    colorbar=dict(title='Error'),
                    showscale=True,
                    opacity=1.0
                ),
                name='Error Points',
                hoverinfo='none'
            ))
        
        # add the RGB-cube edges
        cube_edges = [
            [[0,0,0], [255,0,0]], [[0,0,0], [0,255,0]], [[0,0,0], [0,0,255]],
            [[255,0,0], [255,255,0]], [[255,0,0], [255,0,255]],
            [[0,255,0], [255,255,0]], [[0,255,0], [0,255,255]],
            [[0,0,255], [255,0,255]], [[0,0,255], [0,255,255]],
            [[255,255,0], [255,255,255]], [[255,0,255], [255,255,255]],
            [[0,255,255], [255,255,255]]
        ]
        
        for edge in cube_edges:
            fig.add_trace(go.Scatter3d(
                x=[edge[0][0], edge[1][0]],
                y=[edge[0][1], edge[1][1]],
                z=[edge[0][2], edge[1][2]],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False,
                hoverinfo='none'
            ))
        
        # update the layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Red',
                yaxis_title='Green',
                zaxis_title='Blue',
                xaxis=dict(range=[0, 255]),
                yaxis=dict(range=[0, 255]),
                zaxis=dict(range=[0, 255]),
                aspectmode='cube'
            ),
            width=1000,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
        return fig
    
    def visualize_2d_projections(self, ellipsoids, error_points=None, error_values=None,
                                projections=[('R', 'G'), ('R', 'B'), ('G', 'B')],
                                save_path=None):
        """
        2D projections showing the ellipses
        """
        n_proj = len(projections)
        fig, axes = plt.subplots(1, n_proj, figsize=(6*n_proj, 5))
        
        if n_proj == 1:
            axes = [axes]
        
        axis_map = {'R': 0, 'G': 1, 'B': 2}
        
        for idx, (proj) in enumerate(projections):
            ax = axes[idx]
            x_axis, y_axis = proj
            xi, yi = axis_map[x_axis], axis_map[y_axis]
            
            # plot the error points (if given)
            if error_points is not None and error_values is not None:
                scatter = ax.scatter(error_points[:, xi], error_points[:, yi],
                                    c=error_values, cmap='viridis', s=1, alpha=1.0,
                                    vmin=0, vmax=np.percentile(error_values, 95) if len(error_values) > 0 else 50)
                if idx == 2:  # colour bar only on the last subplot
                    plt.colorbar(scatter, ax=ax, label='Error', shrink=0.8)
            
            # draw every Gaussian's ellipse
            for ellipsoid in ellipsoids:
                center = ellipsoid['center']
                axes_lengths = ellipsoid['axes_lengths']
                rotation = ellipsoid['rotation']
                opacity = ellipsoid['opacity']
                
                # 2x2 covariance sub-matrix for this projection
                cov_2d = np.array([
                    [ellipsoid['covariance'][xi, xi], ellipsoid['covariance'][xi, yi]],
                    [ellipsoid['covariance'][yi, xi], ellipsoid['covariance'][yi, yi]]
                ])
                
                # 2D ellipse parameters
                try:
                    eigvals, eigvecs = np.linalg.eigh(cov_2d)
                    eigvals = np.maximum(eigvals, 1e-6)
                    
                    # ellipse parameters
                    width = 2 * np.sqrt(eigvals[1])  # major axis
                    height = 2 * np.sqrt(eigvals[0])  # minor axis
                    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
                    
                    # build the ellipse
                    ellipse = Ellipse(
                        xy=(center[xi], center[yi]),
                        width=width,
                        height=height,
                        angle=angle,
                        facecolor='none',
                        edgecolor='blue',
                        alpha=opacity,
                        linewidth=2 / (opacity + 0.1)  # lower opacity -> thinner line
                    )
                    ax.add_patch(ellipse)
                    
                    # plot the centre point
                    ax.scatter(center[xi], center[yi], 
                              color='red', s=10 * opacity + 5, alpha=opacity)
                    
                except:
                    # if the covariance is degenerate, draw a circle
                    radius = np.sqrt(ellipsoid['eigenvalues'].max()) * 2
                    circle = plt.Circle(
                        (center[xi], center[yi]),
                        radius=radius,
                        facecolor='none',
                        edgecolor='blue',
                        alpha=opacity
                    )
                    ax.add_patch(circle)
                    ax.scatter(center[xi], center[yi], 
                              color='red', s=10 * opacity + 5, alpha=opacity)
            
            ax.set_xlabel(f'{x_axis}')
            ax.set_ylabel(f'{y_axis}')
            ax.set_title(f'{x_axis}-{y_axis} Projection')
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.set_aspect('equal')
            ax.grid(True, alpha=1.0)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig, axes
    
    def visualize_opacity_contours(self, ellipsoids, resolution=100, save_path=None):
        """
        Opacity contour plot
        """
        # build the grid
        x = np.linspace(0, 255, resolution)
        y = np.linspace(0, 255, resolution)
        X, Y = np.meshgrid(x, y)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # one plot per Z plane
        z_planes = [64, 128, 192]
        
        for idx, z_plane in enumerate(z_planes):
            ax = axes[idx // 2, idx % 2]
            
            # opacity contribution at each grid point
            opacity_map = np.zeros_like(X)
            
            for ellipsoid in ellipsoids:
                center = ellipsoid['center']
                cov = ellipsoid['covariance']
                opacity = ellipsoid['opacity']
                
                # evaluate every Gaussian at each grid point
                for i in range(resolution):
                    for j in range(resolution):
                        point = np.array([X[i, j], Y[i, j], z_plane])
                        diff = point - center
                        
                        try:
                            inv_cov = np.linalg.inv(cov + np.eye(3)*1e-6)
                            mahal_dist = diff @ inv_cov @ diff
                            gauss_val = np.exp(-0.5 * mahal_dist)
                            opacity_map[i, j] += gauss_val * opacity
                        except:
                            pass
            
            # draw the contours
            if opacity_map.max() > 0:
                contour = ax.contour(X, Y, opacity_map, levels=10, cmap='viridis', alpha=1.0)
                ax.clabel(contour, inline=True, fontsize=8)
            
            # plot the Gaussian centres
            for ellipsoid in ellipsoids:
                if abs(ellipsoid['center'][2] - z_plane) < 50:  # only Gaussians near this plane
                    ax.scatter(ellipsoid['center'][0], ellipsoid['center'][1],
                              color='red', s=20, alpha=1.0)
            
            ax.set_xlabel('Red')
            ax.set_ylabel('Green')
            ax.set_title(f'Opacity Contours at Blue={z_plane}')
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.set_aspect('equal')
        
        # 4th panel: opacity vs. position
        ax4 = axes[1, 1]
        
        # sample a few points
        np.random.seed(42)
        n_samples = 1000
        sample_points = np.random.rand(n_samples, 3) * 255
        
        # opacity at each point
        sample_opacities = []
        for point in sample_points:
            total_opacity = 0
            for ellipsoid in ellipsoids:
                diff = point - ellipsoid['center']
                try:
                    inv_cov = np.linalg.inv(ellipsoid['covariance'] + np.eye(3)*1e-6)
                    mahal_dist = diff @ inv_cov @ diff
                    gauss_val = np.exp(-0.5 * mahal_dist)
                    total_opacity += gauss_val * ellipsoid['opacity']
                except:
                    pass
            sample_opacities.append(total_opacity)
        
        sample_opacities = np.array(sample_opacities)
        
        if len(sample_opacities) > 0:
            scatter = ax4.scatter(sample_points[:, 0], sample_points[:, 1],
                                c=sample_opacities, cmap='viridis', s=5, alpha=1.0)
            plt.colorbar(scatter, ax=ax4, label='Total Opacity')
        
        ax4.set_xlabel('Red')
        ax4.set_ylabel('Green')
        ax4.set_title('Opacity Distribution (RG plane, averaged over B)')
        ax4.set_xlim(0, 255)
        ax4.set_ylim(0, 255)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig, axes
    
    def create_animation(self, ellipsoids, error_points=None, error_values=None,
                        save_path='gaussian_rotation.gif'):
        """
        Rotating animation (needs matplotlib.animation)
        """
        from matplotlib.animation import FuncAnimation
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            
            # set the view angle
            ax.view_init(elev=20, azim=frame)
            
            # plot the error points
            if error_points is not None and error_values is not None:
                ax.scatter(error_points[:, 0], error_points[:, 1], error_points[:, 2],
                          c=error_values, cmap='viridis', s=1, alpha=1.0, vmin=0, vmax=50)
            
            # draw the Gaussian ellipsoids
            for ellipsoid in ellipsoids:
                opacity = ellipsoid['opacity']
                if opacity > 0.1:  # only Gaussians with opacity > 0.1
                    self.plot_3d_ellipsoid_matplotlib(ax, ellipsoid, color='blue',
                                                     alpha=0.2, wireframe=True,
                                                     draw_axes=False)
            
            # axis labels / limits
            ax.set_xlabel('Red')
            ax.set_ylabel('Green')
            ax.set_zlabel('Blue')
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.set_zlim(0, 255)
            ax.set_title(f'Gaussian Distributions (Angle: {frame}°)')
        
        anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 5), 
                           interval=100, blit=False)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
        
        plt.show()
        
        return anim
    
    def comprehensive_visualization(self, cube_file_path, num_test_points=5000, save_prefix=None):
        """
        Full visualization report
        """
        
        base_viz = GaussianLUT3DVisualizer(self.model)
        test_points = base_viz.generate_test_points(num_test_points, strategy='uniform')
        # errors_dict = base_viz.compute_errors(test_points)
        errors_dict = base_viz.compute_errors_with_cube(test_points, cube_file_path, interpolation='trilinear')
        
        # read the Gaussian ellipsoids
        ellipsoids = self.get_gaussian_ellipsoids()
        
        print("=" * 80)
        print("Extended Gaussian LUT visualization report")
        print("=" * 80)
        print(f"number of Gaussians: {len(ellipsoids)}")
        print(f"mean opacity: {np.mean([e['opacity'] for e in ellipsoids]):.3f}")
        print(f"mean ellipsoid volume: {np.mean([np.prod(e['axes_lengths']) for e in ellipsoids]):.1f}")
        print()
        
        # # 1. 3D matplotlib visualization (point-cloud mode, more robust)
        # print("3D matplotlib visualization (point-cloud mode)...")
        # self.visualize_3d_matplotlib(
        #     ellipsoids,
        #     error_points=errors_dict['points'],
        #     error_values=errors_dict['euclidean_dist'],
        #     title="Gaussian Distributions with Error Points",
        #     save_path=os.path.join(self.save_dir, f"{save_prefix}_3d_matplotlib.png") if save_prefix else None,
        #     use_points=False  # point-cloud mode is more robust
        # )
        
        # # 2. 2D projection
        # print("2D projection visualization...")
        # self.visualize_2d_projections(
        #     ellipsoids,
        #     error_points=errors_dict['points'],
        #     error_values=errors_dict['euclidean_dist'],
        #     save_path=os.path.join(self.save_dir, f"{save_prefix}_2d_projections.png") if save_prefix else None
        # )
        
        # # 3. opacity contour
        # print("opacity contour plot...")
        # self.visualize_opacity_contours(
        #     ellipsoids,
        #     save_path=os.path.join(self.save_dir, f"{save_prefix}_contours.png") if save_prefix else None
        # )
        
        # 4. interactive plotly visualization
        print("interactive plotly visualization...")
        print(f"max value of error: {errors_dict['delta_e'].max()}")
        self.visualize_3d_plotly(
            ellipsoids,
            error_points=errors_dict['points'],
            error_values=errors_dict['delta_e'],
            title="Interactive Gaussian Distributions",
            save_path=os.path.join(self.save_dir, f"{save_prefix}_interactive.html") if save_prefix else None
        )
        
        return ellipsoids, errors_dict
    
    def visualization_nilut(self, cube_file_path, num_test_points=5000, save_prefix=None):
        """
        Full visualization report
        """
        
        base_viz = GaussianLUT3DVisualizer(self.model)
        test_points = base_viz.generate_test_points(num_test_points, strategy='uniform')
        # errors_dict = base_viz.compute_errors(test_points)
        errors_dict = base_viz.compute_errors_with_cube(test_points, cube_file_path, interpolation='trilinear')
        
        
        # 4. interactive plotly visualization
        print("interactive plotly visualization...")
        print(f"max value of error: {errors_dict['delta_e'].max()}")
        self.visualize_3d_plotly_o_error(
            error_points=errors_dict['points'],
            error_values=errors_dict['delta_e'],
            title="Interactive NILUT Error Distributions",
            save_path=os.path.join(self.save_dir, f"{save_prefix}_interactive.html") if save_prefix else None
        )
        
        return errors_dict


# Quick example
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='NILUT fitting')

    parser.add_argument("--pretrained_model", help="path of pretrained model", 
                        default="./pretrained_models/glut/7luts/GLUT_LUT01_After_Effects_LUTs_Contrast_32_psnr52.64_dE0.3678.pth", type=str)
    parser.add_argument("--cube_file_path", help="path of cube file", 
                        default="./dataset/cube_files/7luts/LUT01_After Effects LUTs_Contrast.cube", type=str)
    parser.add_argument("--save_dir", help="Input RGB map as a hald image", default="./results/vis_glut", type=str)
    parser.add_argument("--num_gaussians", help="gaussian splatting factor: number of gaussians", default=32, type=int)
    parser.add_argument("--random_seed", help="random seed", default=52, type=int)


    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    from train_glut import GLUT3D as GLUT3D3D
    model = GLUT3D3D(num_gaussians=args.num_gaussians, 
                                  residual=True,
                                  logger=None).cuda()
    
    checkpoint = torch.load(args.pretrained_model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    # build the extended visualizer
    viz = EnhancedGaussianLUTVisualizer(model)
    
    # run the full visualization
    ellipsoids, errors = viz.comprehensive_visualization(
        cube_file_path = args.cube_file_path,
        num_test_points=10000,
        save_prefix=os.path.basename(args.pretrained_model)[:-4]
    )
    


    # from nilut import NILUT
    # model = NILUT(in_features=3, hidden_features=128, hidden_layers=2, out_features=3, res=True).cuda()
    # model.load_state_dict(torch.load('3dlut.pt', map_location='cpu'))
    # model.eval()

    # viz = EnhancedGaussianLUTVisualizer(model)
    
    # # run the full visualization
    # errors_dict = viz.visualization_nilut(
    #     cube_file_path = args.cube_file_path,
    #     num_test_points=10000,
    #     save_prefix=os.path.basename(args.cube_file_path)[:-4]
    # )
    
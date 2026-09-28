import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Optional, Tuple, List
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import setup_logging
from data.apply_cube import CubeFileTransform
from train_glut import GLUT3D as ContinuousGaussianLUT3D


class Arrow3D(FancyArrowPatch):
    """Arrow artist for 3D plots"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class LUTColorTransformVisualizer:
    """
    Visualize how the LUT transforms colours
    """
    def __init__(self, 
                 model,  # Gaussian LUT model
                 cube_transform=None,  # optional .cube transform
                 save_dir="./results/vis",
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.cube_transform = cube_transform
        self.save_dir = save_dir
        self.device = device
        self.model.eval()
    
    def generate_sample_points(self, 
                              num_points: int = 1000,
                              strategy: str = 'grid') -> np.ndarray:
        """
        Generate sample points
        
        Args:
            num_points: sample points
            strategy: 'grid' | 'random' | 'stratified'
        
        Returns:
            points: [N, 3] RGB points in [0, 1]
        """
        if strategy == 'grid':
            # grid sampling
            n_per_dim = int(round(num_points ** (1/3)))
            x = np.linspace(0, 1, n_per_dim)
            y = np.linspace(0, 1, n_per_dim)
            z = np.linspace(0, 1, n_per_dim)
            X, Y, Z = np.meshgrid(x, y, z)
            points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
            
            # sub-sample if there are too many points
            if len(points) > num_points:
                indices = np.random.choice(len(points), num_points, replace=False)
                points = points[indices]
                
        elif strategy == 'stratified':
            # stratified sampling (covers the whole cube)
            n_per_dim = int(round(num_points ** (1/3)))
            points = []
            for i in range(n_per_dim):
                for j in range(n_per_dim):
                    for k in range(n_per_dim):
                        # one random sample per sub-cube
                        p = np.array([
                            np.random.uniform(i/n_per_dim, (i+1)/n_per_dim),
                            np.random.uniform(j/n_per_dim, (j+1)/n_per_dim),
                            np.random.uniform(k/n_per_dim, (k+1)/n_per_dim)
                        ])
                        points.append(p)
            points = np.array(points[:num_points])
            
        else:  # random
            points = np.random.rand(num_points, 3)
        
        return points
    
    def compute_color_transform(self, 
                               points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute colours before / after the transform
        
        Returns:
            original_colors: original colours (used to colour the points)
            transformed_colors: transformed colours
        """
        with torch.no_grad():
            points_tensor = torch.FloatTensor(points).to(self.device)
            
            # model prediction
            if hasattr(self.model, 'forward'):
                model_transformed = self.model(points_tensor)
            else:
                model_transformed = self.model(points_tensor)
            
            model_transformed = model_transformed.cpu().numpy()
            
            # also compute the .cube transform if a file was given
            if self.cube_transform is not None:
                cube_transformed = self.cube_transform(points_tensor)
                cube_transformed = cube_transformed.cpu().numpy()
                return model_transformed, cube_transformed
            
            return model_transformed, None
    
    def visualize_3d_arrows_matplotlib(self,
                                      points: np.ndarray,
                                      transformed_colors: np.ndarray,
                                      title: str = "LUT Color Transformation",
                                      arrow_scale: float = 1.0,
                                      save_path: Optional[str] = None):
        """
        3D quiver plot (matplotlib)
        """
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # plot the original points (coloured by their RGB)
        scatter = ax.scatter(points[:, 0] * 255, points[:, 1] * 255, points[:, 2] * 255,
                           c=points, s=10, alpha=0.6, label='Original')
        
        # plot the transformed points
        ax.scatter(transformed_colors[:, 0] * 255, 
                  transformed_colors[:, 1] * 255, 
                  transformed_colors[:, 2] * 255,
                  c=transformed_colors, s=7, alpha=0.3, label='Transformed')
        
        # plot the arrows
        for i in range(len(points)):
            start = points[i] * 255
            end = transformed_colors[i] * 255
            
            arrow = Arrow3D([start[0], end[0]],
                            [start[1], end[1]],
                            [start[2], end[2]],
                            mutation_scale=10,
                            lw=1,
                            arrowstyle='-|>',
                            color=transformed_colors[i],
                            alpha=0.5)
            ax.add_artist(arrow)
        
        # draw the RGB-cube edges
        cube_edges = [
            [[0,0,0], [255,0,0]], [[0,0,0], [0,255,0]], [[0,0,0], [0,0,255]],
            [[255,0,0], [255,255,0]], [[255,0,0], [255,0,255]],
            [[0,255,0], [255,255,0]], [[0,255,0], [0,255,255]],
            [[0,0,255], [255,0,255]], [[0,0,255], [0,255,255]],
            [[255,255,0], [255,255,255]], [[255,0,255], [255,255,255]],
            [[0,255,255], [255,255,255]]
        ]
        
        for edge in cube_edges:
            ax.plot([edge[0][0], edge[1][0]],
                   [edge[0][1], edge[1][1]],
                   [edge[0][2], edge[1][2]],
                   'gray', linewidth=0.5, alpha=0.3)
        
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_title(title)
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)
        ax.legend()
        
        # equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig, ax
    
    def visualize_3d_arrows_plotly(self,
                                  points: np.ndarray,
                                  transformed_colors: np.ndarray,
                                  title: str = "LUT Color Transformation",
                                  save_path: Optional[str] = None):
        """
        Interactive 3D quiver plot (plotly)
        """
        
        fig = go.Figure()
        
        # add the original points
        fig.add_trace(go.Scatter3d(
            x=points[:, 0] * 255,
            y=points[:, 1] * 255,
            z=points[:, 2] * 255,
            mode='markers',
            marker=dict(
                size=4,
                color=points,
                opacity=0.6
            ),
            name='Original Points',
            hoverinfo='text',
            text=[f'RGB: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})' for p in points]
        ))
        
        # add the transformed points
        fig.add_trace(go.Scatter3d(
            x=transformed_colors[:, 0] * 255,
            y=transformed_colors[:, 1] * 255,
            z=transformed_colors[:, 2] * 255,
            mode='markers',
            marker=dict(
                size=4,
                color=transformed_colors,
                opacity=1.0
            ),
            name='Transformed Points',
            hoverinfo='text',
            text=[f'RGB: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})' for p in transformed_colors]
        ))
        
        # add the arrows
        arrow_traces = []
        for i in range(len(points)):
            start = points[i] * 255
            end = transformed_colors[i] * 255

            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
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
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
        return fig
    
    def visualize_2d_projections(self,
                                points: np.ndarray,
                                transformed_colors: np.ndarray,
                                projections: List[Tuple[str, str]] = [('R', 'G'), ('R', 'B'), ('G', 'B')],
                                save_path: Optional[str] = None):
        """
        2D projections showing the colour transform
        """
        axis_map = {'R': 0, 'G': 1, 'B': 2}
        n_proj = len(projections)
        
        fig, axes = plt.subplots(1, n_proj, figsize=(6*n_proj, 5))
        if n_proj == 1:
            axes = [axes]
        
        for idx, (proj) in enumerate(projections):
            ax = axes[idx]
            x_axis, y_axis = proj
            xi, yi = axis_map[x_axis], axis_map[y_axis]
            
            # plot the original points
            ax.scatter(points[:, xi] * 255, points[:, yi] * 255,
                      c=points, s=5, alpha=0.5, label='Original')
            
            # plot the transformed points
            ax.scatter(transformed_colors[:, xi] * 255, 
                      transformed_colors[:, yi] * 255,
                      c=transformed_colors, s=5, alpha=0.7, label='Transformed')
            
            # plot a few arrows
            for i in range(len(points)):
                start = points[i] * 255
                end = transformed_colors[i] * 255
                
                ax.arrow(start[xi], start[yi],
                        end[0] - start[xi], end[1] - start[yi],
                        head_width=3, head_length=3, fc=transformed_colors[i], ec=transformed_colors[i],
                        alpha=0.5, length_includes_head=True)
            
            ax.set_xlabel(f'{x_axis}')
            ax.set_ylabel(f'{y_axis}')
            ax.set_title(f'{x_axis}-{y_axis} Projection')
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig, axes
    
    def visualize_color_mapping(self,
                               points: np.ndarray,
                               transformed_colors: np.ndarray,
                               save_path: Optional[str] = None):
        """
        Visualize the colour mapping (with colour bars)
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. original-colour distribution
        ax = axes[0, 0]
        ax.hist(points[:, 0], bins=50, alpha=0.5, color='red', label='R')
        ax.hist(points[:, 1], bins=50, alpha=0.5, color='green', label='G')
        ax.hist(points[:, 2], bins=50, alpha=0.5, color='blue', label='B')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Original Color Distribution')
        ax.legend()
        
        # 2. transformed-colour distribution
        ax = axes[0, 1]
        ax.hist(transformed_colors[:, 0], bins=50, alpha=0.5, color='red', label='R')
        ax.hist(transformed_colors[:, 1], bins=50, alpha=0.5, color='green', label='G')
        ax.hist(transformed_colors[:, 2], bins=50, alpha=0.5, color='blue', label='B')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Transformed Color Distribution')
        ax.legend()
        
        # 3. transform-vector-length distribution
        ax = axes[0, 2]
        diff = transformed_colors - points
        distances = np.linalg.norm(diff, axis=1)
        ax.hist(distances, bins=50, alpha=0.7, color='purple')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Frequency')
        ax.set_title('Transformation Distance Distribution')
        
        # 4. original vs. transformed scatter
        ax = axes[1, 0]
        ax.scatter(points[:, 0], transformed_colors[:, 0], 
                  alpha=0.1, s=1, color='red')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('Original R')
        ax.set_ylabel('Transformed R')
        ax.set_title('R Channel Mapping')
        
        ax = axes[1, 1]
        ax.scatter(points[:, 1], transformed_colors[:, 1], 
                  alpha=0.1, s=1, color='green')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('Original G')
        ax.set_ylabel('Transformed G')
        ax.set_title('G Channel Mapping')
        
        ax = axes[1, 2]
        ax.scatter(points[:, 2], transformed_colors[:, 2], 
                  alpha=0.1, s=1, color='blue')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('Original B')
        ax.set_ylabel('Transformed B')
        ax.set_title('B Channel Mapping')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig, axes
    
    def comprehensive_visualization(self,
                                   num_points: int = 2000,
                                   strategy: str = 'stratified',
                                   save_prefix: Optional[str] = None):
        """
        Full visualization report
        """
        # Generate sample points
        points = self.generate_sample_points(num_points, strategy)
        
        # compute the transform
        model_transformed, cube_transformed = self.compute_color_transform(points)
        
        print("=" * 80)
        print("LUT colour-transform visualization report")
        print("=" * 80)
        print(f"sample points: {len(points)}")
        print(f"sampling strategy: {strategy}")

        for transformed, trans in zip([model_transformed, cube_transformed], ["pred", "target"]):
        
            # compute statistics
            diff = transformed - points
            distances = np.linalg.norm(diff, axis=1)
            
            print(f"\ntransform stats:")
            print(f"  mean transform distance: {distances.mean():.4f}")
            print(f"  max transform distance: {distances.max():.4f}")
            print(f"  min transform distance: {distances.min():.4f}")
            print(f"  transform distance std: {distances.std():.4f}")
            
            print(f"\nmean per-channel change:")
            for i, channel in enumerate(['R', 'G', 'B']):
                mean_change = np.abs(diff[:, i]).mean()
                print(f"  {channel}: {mean_change:.4f}")
            
            # 1. 3D matplotlib visualization
            print("\n3D matplotlib visualization...")
            self.visualize_3d_arrows_matplotlib(
                points, transformed,
                title="LUT Color Transformation (Matplotlib)",
                save_path=os.path.join(self.save_dir, f"{save_prefix}_3d_{trans}.png" if save_prefix else None)
            )
            
            # # 2. 2D projection
            # print("2D projection visualization...")
            # self.visualize_2d_projections(
            #     points, transformed,
            #     save_path=os.path.join(self.save_dir, f"{save_prefix}_2d_{trans}.png" if save_prefix else None)
            # )
            
            # 3. colour-mapping analysis
            print("colour-mapping analysis...")
            self.visualize_color_mapping(
                points, transformed,
                save_path=os.path.join(self.save_dir, f"{save_prefix}_mapping_{trans}.png" if save_prefix else None)
            )
            
            # 4. interactive plotly visualization
            print("interactive plotly visualization...")
            self.visualize_3d_arrows_plotly(
                points, transformed,
                title="LUT Color Transformation (Interactive)",
                save_path=os.path.join(self.save_dir, f"{save_prefix}_{trans}.html" if save_prefix else None)
            )
            
            # # 5. compare model vs. cube if a .cube file is given
            # if cube_transformed is not None:
            #     print("\nmodel vs. cube comparison...")
            #     self.visualize_model_vs_cube(
            #         points, transformed, cube_transformed,
            #         save_path=f"{save_prefix}_comparison.png" if save_prefix else None
            #     )
        
    
    def visualize_model_vs_cube(self,
                               points: np.ndarray,
                               model_transformed: np.ndarray,
                               cube_transformed: np.ndarray,
                               save_path: Optional[str] = None):
        """
        Compare the model output against the .cube transform
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        diff = model_transformed - cube_transformed
        distances = np.linalg.norm(diff, axis=1)
        
        # 1. difference distribution
        ax = axes[0, 0]
        ax.hist(distances, bins=50, alpha=0.7)
        ax.set_xlabel('Difference')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Model vs Cube Difference\nMean: {distances.mean():.4f}')
        
        # 2. per-channel difference
        ax = axes[0, 1]
        for i, (channel, color) in enumerate(zip(['R', 'G', 'B'], ['red', 'green', 'blue'])):
            ax.hist(diff[:, i], bins=50, alpha=0.5, color=color, label=channel)
        ax.set_xlabel('Channel Difference')
        ax.set_ylabel('Frequency')
        ax.set_title('Per-Channel Differences')
        ax.legend()
        
        # 3. difference vs. luminance
        ax = axes[0, 2]
        brightness = points.mean(axis=1)
        ax.scatter(brightness, distances, alpha=0.3, s=1)
        ax.set_xlabel('Brightness')
        ax.set_ylabel('Difference')
        ax.set_title('Difference vs Brightness')
        
        # 4. difference in colour space (RG projection)
        ax = axes[1, 0]
        scatter = ax.scatter(points[:, 0] * 255, points[:, 1] * 255,
                            c=distances, cmap='hot', s=2, alpha=0.5)
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_title('Difference Distribution (RG)')
        plt.colorbar(scatter, ax=ax)
        
        # 5. difference in colour space (RB projection)
        ax = axes[1, 1]
        scatter = ax.scatter(points[:, 0] * 255, points[:, 2] * 255,
                            c=distances, cmap='hot', s=2, alpha=0.5)
        ax.set_xlabel('R')
        ax.set_ylabel('B')
        ax.set_title('Difference Distribution (RB)')
        plt.colorbar(scatter, ax=ax)
        
        # 6. difference in colour space (GB projection)
        ax = axes[1, 2]
        scatter = ax.scatter(points[:, 1] * 255, points[:, 2] * 255,
                            c=distances, cmap='hot', s=2, alpha=0.5)
        ax.set_xlabel('G')
        ax.set_ylabel('B')
        ax.set_title('Difference Distribution (GB)')
        plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig, axes


# Example
if __name__ == "__main__":
    # assumes you have a trained model

    parser = argparse.ArgumentParser(description='GLUT visualization')
    parser.add_argument("--pretrained_model", help="path of pretrained model", 
                        default="./pretrained_models/glut/7luts/GLUT_LUT01_After_Effects_LUTs_Contrast_32_psnr52.64_dE0.3678.pth", type=str)
    parser.add_argument("--cube_file_path", help="path of cube file", 
                        default="./dataset/cube_files/7luts/LUT01_After Effects LUTs_Contrast.cube", type=str)
    parser.add_argument("--save_dir", help="Input RGB map as a hald image", default="./results/vis_glut", type=str)
    parser.add_argument("--num_gaussians", help="gaussian splatting factor: number of gaussians", default=32, type=int)
    parser.add_argument("--random_seed", help="random seed", default=52, type=int)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    model = ContinuousGaussianLUT3D(num_gaussians=args.num_gaussians, 
                                  residual=True,
                                  logger=None).cuda()
    checkpoint = torch.load(args.pretrained_model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    cube_transform = CubeFileTransform(args.cube_file_path, interpolation='trilinear')

    viz = LUTColorTransformVisualizer(model, cube_transform, args.save_dir)
    results = viz.comprehensive_visualization(
        num_points=500,
        strategy='grid',
        save_prefix=os.path.basename(args.pretrained_model)[:-4]
    )

    pass
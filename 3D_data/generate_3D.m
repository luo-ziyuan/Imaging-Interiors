%% save forward data including Profile parameters and scattered field
% Modified from Wei Zhun's 2D version to 3D with polarization effects

clc; clear all; close all;
load Forward_Circ1.mat;   % Make Sure it is the same profile as in Previous traning
clear E_s

%% Basic parameters
eta_0 = 120*pi;          % Free space impedance
c = 3e8;                 % Speed of light
eps_0 = 8.85e-12;        % Vacuum permittivity
freq = 0.4;              % Frequency in GHz
lam_0 = c/(freq*1e9);    % Wavelength
k_0 = 2*pi/lam_0;        % Wavenumber
omega = k_0*c;           % Angular frequency
bool_plane = 1;          % 1: Plane wave incidence; 0: Line source incidence
Coef = i*k_0*eta_0;

% Sampling parameters
N_rec = 32;              % Number of Receivers
N_inc = 8;               % Number of Incidences
N_phi = 5;               % Number of phi angles

% Discretization parameters
MAX = 1; 
Mx = 25;                 % Grid size
step_size = 2*MAX/(Mx-1);
cell_volume = step_size^3;  % Volume of the sub-domain
a_eqv = (cell_volume*3/(4*pi))^(1/3);  % Equivalent radius for 3D


%% Generate computational domain
tmp_domain = linspace(-MAX, MAX, Mx);
[x_dom, y_dom, z_dom] = meshgrid(tmp_domain, -tmp_domain, -tmp_domain);
N_cell_dom = size(x_dom,1)*size(x_dom,2)*size(x_dom,3);
x0 = x_dom; y0 = y_dom; z0 = z_dom;

%% Load or generate scatterer profile
load('./original_data/voxelgrid_sample0.mat');
epsil = voxelgrid(end:-1:1, :, :);
epsil(epsil > 0) = 1.0;
epsil = epsil + 1;
epsil_exaS1 = epsil;

% Calculate contrast function
xi_all = -i*omega*(epsil-1)*eps_0*cell_volume;  
bool_eps = epsil==-1; 
in_anulus = find(bool_eps==0); 
in_anulus = in_anulus(:);

% Remove points outside scatterer
x0(bool_eps) = []; 
y0(bool_eps) = []; 
z0(bool_eps) = [];
x0 = x0(:); y0 = y0(:); z0 = z0(:);
xi_forward = xi_all; 
xi_forward(bool_eps) = []; 
xi_forward = xi_forward(:);
N_cell = length(x0);

%% Set up receivers and calculate unit vectors
theta_tmp = linspace(0, 2*pi, N_rec+1); 
theta_tmp(end) = []; 
theta_tmp = theta_tmp(:);
phi_tmp = linspace(0, pi, N_phi+2);  % phi从0°到180°
phi_tmp(end) = []; 
phi_tmp(1) = []; 
phi_tmp = phi_tmp(:);

[theta, phi, rho] = meshgrid(theta_tmp, phi_tmp, 3); 
theta = theta(:); 
rho = rho(:); 
phi = phi(:);
[x, y, z] = sph2cart(theta, pi/2-phi, rho);  % 注意这里用pi/2-phi

% Calculate receiver polarization vectors
% e_r指向观测点
e_r_rec = [sin(phi).*cos(theta), sin(phi).*sin(theta), cos(phi)];

% e_theta在xoy平面内，垂直于水平方向
e_theta_rec = [-sin(theta), cos(theta), zeros(size(theta))];

% e_phi由叉积得到
e_phi_rec = -cross(e_r_rec, e_theta_rec);

% Visualize receiver positions
figure;
scatter3(x, y, z, [], 'b', 'filled');
xlabel('X'); ylabel('Y'); zlabel('Z');
grid on; axis equal;

% Add polarization vectors visualization
figure;
quiver3(x, y, z, e_theta_rec(:,1), e_theta_rec(:,2), e_theta_rec(:,3), 0.5, 'r', 'DisplayName', 'E_\theta');
hold on;
quiver3(x, y, z, e_phi_rec(:,1), e_phi_rec(:,2), e_phi_rec(:,3), 0.5, 'b', 'DisplayName', 'E_\phi');
scatter3(x, y, z, 'k', 'filled', 'DisplayName', 'Observation Points');
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Receiver Positions and Polarization Directions');
legend;
grid on;
axis equal;


if bool_plane == 1
    %% Plane wave incidence setup
    theta_inc = linspace(0, 2*pi, N_inc+1); 
    theta_inc(end) = []; 
    theta_inc = theta_inc(:);  
    phi_inc = linspace(0, pi, N_phi+2);  % phi从0°到180°
    phi_inc(end) = []; 
    phi_inc(1) = []; 
    phi_inc = phi_inc(:);

    % Source positions and polarization vectors
    [theta_t, phi_t, rho_t] = meshgrid(theta_inc, phi_inc, 3);
    theta_t = theta_t(:); 
    phi_t = phi_t(:); 
    rho_t = rho_t(:);
    [x_t, y_t, z_t] = sph2cart(theta_t, pi/2-phi_t, rho_t);
    xyz_t = [x_t, y_t, z_t];

    % Calculate source polarization vectors
    [theta_inc, phi_inc] = meshgrid(theta_inc, phi_inc);
    theta_inc = theta_inc(:);
    phi_inc = phi_inc(:);

    % 修正发射源的极化向量
    e_r_src = [sin(phi_inc).*cos(theta_inc), sin(phi_inc).*sin(theta_inc), cos(phi_inc)];
    e_theta_src = [-sin(theta_inc), cos(theta_inc), zeros(size(theta_inc))];
    e_phi_src = -cross(e_r_src, e_theta_src);

    % 修正波矢量分量
    k_x = -k_0*sin(phi_inc).*cos(theta_inc);
    k_y = -k_0*sin(phi_inc).*sin(theta_inc);
    k_z = -k_0*cos(phi_inc);

    % Calculate incident field with phi polarization
    E_inc = zeros(N_cell, N_inc, 3);
    for i = 1:N_inc
        phase = exp(1i*(x0*k_x(i) + y0*k_y(i) + z0*k_z(i)));
        E_inc(:,i,:) = phase .* reshape(e_phi_src(i,:), [1,1,3]);
    end

    % Add position distribution visualization
    figure('Position', [100, 100, 1200, 500]);

    % 左边显示接收机位置
    subplot(1,2,1);
    scatter3(x, y, z, 50, phi, 'filled');  % 用颜色表示phi角度
    colorbar;
    title('接收机位置分布');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;
    grid on;
    colormap('jet');
    view(45, 30);  % 设置视角
    % 添加单位球面参考
    [X,Y,Z] = sphere(50);
    hold on;
    surf(X*3, Y*3, Z*3, 'FaceAlpha', 0.1, 'EdgeAlpha', 0.3);  % 3是球半径

    % 右边显示发射源位置
    subplot(1,2,2);
    scatter3(x_t, y_t, z_t, 50, phi_t, 'filled');  % 用颜色表示phi角度
    colorbar;
    title('发射源位置分布');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;
    grid on;
    colormap('jet');
    view(45, 30);  % 设置视角
    % 添加单位球面参考
    hold on;
    surf(X*3, Y*3, Z*3, 'FaceAlpha', 0.1, 'EdgeAlpha', 0.3);  % 3是球半径

    % Visualize source positions and polarization
    figure;
    quiver3(x_t, y_t, z_t, e_theta_src(:,1), e_theta_src(:,2), e_theta_src(:,3), 0.5, 'r', 'DisplayName', 'E_\theta');
    hold on;
    quiver3(x_t, y_t, z_t, e_phi_src(:,1), e_phi_src(:,2), e_phi_src(:,3), 0.5, 'b', 'DisplayName', 'E_\phi');
    scatter3(x_t, y_t, z_t, 'k', 'filled', 'DisplayName', 'Source Points');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('Source Positions and Polarization Directions');
    legend;
    grid on;
    axis equal;
else
    %% Line source incidence setup
    theta_inc = linspace(0, 2*pi, N_inc+1); 
    theta_inc(end) = []; 
    theta_inc = theta_inc(:);  
    phi_inc = linspace(0, pi, N_phi+2);  % phi从0°到180°
    phi_inc(end) = []; 
    phi_inc(1) = []; 
    phi_inc = phi_inc(:);

    % Source positions
    [theta_t, phi_t, rho_t] = meshgrid(theta_inc, phi_inc, 3);
    theta_t = theta_t(:); 
    phi_t = phi_t(:); 
    rho_t = rho_t(:);
    [x_t, y_t, z_t] = sph2cart(theta_t, pi/2-phi_t, rho_t);
    xyz_t = [x_t, y_t, z_t];

    % Calculate source polarization vectors
    [theta_inc, phi_inc] = meshgrid(theta_inc, phi_inc);
    theta_inc = theta_inc(:);
    phi_inc = phi_inc(:);

    % 线源的极化向量
    e_r_src = [sin(phi_inc).*cos(theta_inc), sin(phi_inc).*sin(theta_inc), cos(phi_inc)];
    e_theta_src = [-sin(theta_inc), cos(theta_inc), zeros(size(theta_inc))];
    e_phi_src = -cross(e_r_src, e_theta_src);

    % Calculate incident field for line source
    E_inc = zeros(N_cell, N_inc, 3);
    for i = 1:length(theta_inc)
        for j = 1:N_cell
            % 计算观测点到源点的距离向量
            r_vec = [x0(j)-x_t(i), y0(j)-y_t(i), z0(j)-z_t(i)];
            r = norm(r_vec);
            
            % 计算汉克尔函数（使用贝塞尔函数近似）
            % H0(kr) ≈ J0(kr) - iY0(kr)
            kr = k_0 * r;
            H0 = besselj(0,kr) - 1i*bessely(0,kr);
            
            % 计算入射场（使用phi极化）
            E_inc(j,i,:) = H0/4 * e_phi_src(i,:);
        end
    end

    % Add position distribution visualization
    figure('Position', [100, 100, 1200, 500]);

    % 左边显示接收机位置
    subplot(1,2,1);
    scatter3(x, y, z, 50, phi, 'filled');  % 用颜色表示phi角度
    colorbar;
    title('接收机位置分布');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;
    grid on;
    colormap('jet');
    view(45, 30);  

    [X,Y,Z] = sphere(50);
    hold on;
    surf(X*3, Y*3, Z*3, 'FaceAlpha', 0.1, 'EdgeAlpha', 0.3); 


    subplot(1,2,2);
    scatter3(x_t, y_t, z_t, 50, phi_t, 'filled'); 
    colorbar;
    title('线源位置分布');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;
    grid on;
    colormap('jet');
    view(45, 30);  

    hold on;
    surf(X*3, Y*3, Z*3, 'FaceAlpha', 0.1, 'EdgeAlpha', 0.3);  

    % Visualize source positions and polarization
    figure;
    quiver3(x_t, y_t, z_t, e_theta_src(:,1), e_theta_src(:,2), e_theta_src(:,3), 0.5, 'r', 'DisplayName', 'E_\theta');
    hold on;
    quiver3(x_t, y_t, z_t, e_phi_src(:,1), e_phi_src(:,2), e_phi_src(:,3), 0.5, 'b', 'DisplayName', 'E_\phi');
    scatter3(x_t, y_t, z_t, 'k', 'filled', 'DisplayName', 'Source Points');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('Line Source Positions and Polarization Directions');
    legend;
    grid on;
    axis equal;
end

%% MOM calculations
% Calculate Green's function matrix
Phi_mat = zeros(N_cell);
[x0_cell, x0_cell_2] = meshgrid(x0,x0);
[y0_cell, y0_cell_2] = meshgrid(y0,y0);
[z0_cell, z0_cell_2] = meshgrid(z0,z0);
dist_cell = sqrt((x0_cell-x0_cell_2).^2 + (y0_cell-y0_cell_2).^2 + (z0_cell-z0_cell_2).^2);
clear x0_cell x0_cell_2 y0_cell y0_cell_2 z0_cell z0_cell_2

% 3D Green's function
dist_cell = dist_cell + eye(N_cell);
I1 = exp(1i*k_0*dist_cell)./(4*pi*dist_cell);
Phi_mat = Coef * I1;
Phi_mat = Phi_mat .*(ones(N_cell)-eye(N_cell));

% Self-term correction
I2 = 1/(4*pi*a_eqv) * (1 - 1i*k_0*a_eqv) * exp(1i*k_0*a_eqv);
S1 = Coef * I2;
Phi_mat = Phi_mat + S1*eye(N_cell);
    
% Calculate total field
E_tot = zeros(N_cell, N_inc, 3);
for i = 1:N_inc
    E_tot(:,i,:) = (eye(N_cell)-Phi_mat*diag(xi_forward)) \ squeeze(E_inc(:,i,:));
end

% Calculate observation points Green's function
R_mat = zeros(N_rec, N_cell, 3);
for i = 1:N_rec
    for j = 1:N_cell
        r_vec = [x(i)-x0(j), y(i)-y0(j), z(i)-z0(j)];
        r = norm(r_vec);
        G = exp(1i*k_0*r)/(4*pi*r);
        R_mat(i,j,:) = G * e_phi_rec(i,:);
    end
end

% Calculate scattered field
E_s = zeros(N_rec, N_inc);
for i = 1:N_inc
    for j = 1:N_rec
        R_vec = squeeze(R_mat(j,:,:));  
        E_vec = squeeze(E_tot(:,i,:));  
        E_s(j,i) = sum(sum(R_vec .* E_vec));
    end
end

% Calculate current
J = zeros(N_cell, N_inc, 3);
for i = 1:N_inc
    J(:,i,:) = xi_forward .* squeeze(E_tot(:,i,:));
end


%% Save results
folder = "syn3D_sample0_25X25";
mkdir(folder)

% Save scattered field
writeNPY(real(E_s), folder + '/E_s_real.npy');
writeNPY(imag(E_s), folder + '/E_s_imag.npy');

% Save ground truth permittivity
writeNPY(epsil_exaS1, folder + '/epsilon_gt.npy');

% Save incident field
writeNPY(real(E_inc), folder + '/E_inc_real.npy');
writeNPY(imag(E_inc), folder + '/E_inc_imag.npy');

% Save Green's function matrices
writeNPY(real(Phi_mat), folder + '/Phi_mat_real.npy');
writeNPY(imag(Phi_mat), folder + '/Phi_mat_imag.npy');
writeNPY(real(R_mat), folder + '/R_mat_real.npy');
writeNPY(imag(R_mat), folder + '/R_mat_imag.npy');

% Save geometry information
writeNPY(xyz_t, folder + '/xyz_t.npy');
writeNPY(x_dom, folder + '/x_dom.npy');
writeNPY(y_dom, folder + '/y_dom.npy');
writeNPY(z_dom, folder + '/z_dom.npy');

% Save total field and current
writeNPY(real(E_tot), folder + '/E_tot_real.npy');
writeNPY(imag(E_tot), folder + '/E_tot_imag.npy');
writeNPY(real(J), folder + '/J_real.npy');
writeNPY(imag(J), folder + '/J_imag.npy');

clearvars -except E_s epsil_exaS1;
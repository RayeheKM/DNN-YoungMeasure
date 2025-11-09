% ========================================================================
% MATLAB Script: Publication-Ready Figures for 1D PINN Results
% Professional style, exact PDF size, high resolution
% ========================================================================

close all; clear; clc;

% Set default font and axes style
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 12);

% ---------------------------------
% 1. Loss History Plot
% ---------------------------------
loss_file = dir('1D3_loss_data.mat');
if ~isempty(loss_file)
    S = load(loss_file(1).name); % loss_history, term1_history, term2_history, term3_history, bc_history

    fig = figure('Units','inches','Position',[0 0 4.3 2]);
    hold on;
    plot(S.loss_history, 'Color',[0 0.45 0.74],'LineWidth',1.5);
    plot(S.term1_history, 'Color',[0.85 0.33 0.10],'LineWidth',1, 'LineStyle', '--');
    plot(S.bc_history, 'Color',[0.47 0.67 0.19],'LineWidth',1, 'LineStyle', '--');
    set(gca,'FontName','Times New Roman');
    xlabel('Epoch');
    ylabel('Loss');
    set(gca,'YScale','log');
    grid off;
    box on;
    legend({'Total','|Energy|^2','|BC|^2'}, 'Location', 'northeast', 'NumColumns', 3);
    set(fig,'PaperUnits','inches','PaperPositionMode','auto');
    box on;
    exportgraphics(fig,'1D3_loss.pdf','ContentType','vector','Resolution',600);
    close(fig);
end

%%
% ---------------------------------
% 2. F(x,z) Surface
% ---------------------------------
surface_file = dir('1D3_3d_F_data.mat');
if ~isempty(surface_file)
    S = load(surface_file(1).name); % x, z, F
    fig = figure('Units','inches','Position',[0 0 2.5 1.9]);
    surf(S.x, S.z, S.F, 'EdgeColor','none');
    colormap(viridis); 
    % shading interp;
    % camlight headlight; lighting phong;
    xlabel('x'); ylabel('\xi'); zlabel('F');
    view(30,10); grid on;
    set(gca,'FontName','Times New Roman');
    set(fig,'PaperUnits','inches','PaperPositionMode','auto');
    box on;
    exportgraphics(fig, '1D3_3d_F.pdf','ContentType','image','Resolution',600); % Use 'image' for surface
    close(fig);
end
%%
% ---------------------------------
% 3. dF/dz Surface
% ---------------------------------
dfdz_file = dir('1D3_3d_dFdz_data.mat');
if ~isempty(dfdz_file)
    S = load(dfdz_file(1).name); % x, z, dFdz
    fig = figure('Units','inches','Position',[0 0 2.5 1.9]);
    surf(S.x, S.z, S.dFdz, 'EdgeColor','none');
    colormap(summer); shading interp;
    camlight headlight; lighting phong;
    xlabel('x'); ylabel('\xi'); zlabel('\partial F/\partial \xi');
    view(120,30); grid on;
    set(gca,'FontName','Times New Roman');
    set(fig,'PaperUnits','inches','PaperPositionMode','auto');
    box on;
    exportgraphics(fig, '1D3_3d_dFdz.pdf','ContentType','image','Resolution',600); % Use 'image' for surface
    close(fig);
end

%%
% ---------------------------------
% 4. F(z) for Fixed x
% ---------------------------------
x_values = [0.25 0.5 0.75];
colors = [0 0.45 0.74; 0.85 0.33 0.10; 0.47 0.67 0.19];

fig = figure('Units','inches','Position',[0 0 2.5 1.9]);
hold on;
for i = 1:numel(x_values)
    file = dir(sprintf('1D3_F_x%g_data.mat', x_values(i)));
    if isempty(file), continue, end
    S = load(file(1).name); % z, F
    plot(S.z, S.F, 'LineWidth', 1.5, 'Color', colors(i,:));
end
xlabel('\xi'); ylabel('F');
set(gca,'FontName','Times New Roman');
grid off;
box on;
legend(arrayfun(@(x) sprintf('x = %.2f', x), x_values, 'UniformOutput', false), 'Location', 'southeast', 'NumColumns', 3);
set(fig,'PaperUnits','inches','PaperPositionMode','auto');
box on;
exportgraphics(fig, '1D3_F_vs_z.pdf', 'ContentType', 'vector', 'Resolution', 600);
close(fig);

%%
% ---------------------------------
% 5. dF/dz(z) for Fixed x
% ---------------------------------
fig = figure('Units','inches','Position',[0 0 4.3 2]);
hold on;
for i = 1:numel(x_values)
    file = dir(sprintf('1D3_dFdz_x%g_data.mat', x_values(i)));
    if isempty(file), continue, end
    S = load(file(1).name); % z, dFdz
    plot(S.z, S.dFdz, 'LineWidth', 1.5, 'Color', colors(i,:));
end
xlabel('\xi'); ylabel('\partial F/\partial \xi');
set(gca,'FontName','Times New Roman');
grid off;
box on;
legend(arrayfun(@(x) sprintf('x = %.2f', x), x_values, 'UniformOutput', false), 'Location','northeast');
set(fig,'PaperUnits','inches','PaperPositionMode','auto');
box on;
exportgraphics(fig, '1D3_dFdz_vs_z.pdf', 'ContentType', 'vector', 'Resolution', 600);
close(fig);

%%
% ---------------------------------
% 6. Histogram of dF/dz
% ---------------------------------
hist_file = dir('1D3_histogram_dFdz_data.mat');
if ~isempty(hist_file)
    S = load(hist_file(1).name); % dFdz_values
    fig = figure('Units','inches','Position',[0 0 2.3 1.6]);
    histogram(S.dFdz_values, 10, 'FaceColor', [0.85 0.33 0.10], 'FaceAlpha', 0.7, 'Normalization', 'pdf', 'EdgeColor', 'none');
    xlabel('\partial F/\partial \xi'); ylabel('Density');
    xlim([-1.5 1.5]);
    set(gca,'FontName','Times New Roman','GridLineStyle','--','GridAlpha',0.3);
    grid off; box on
    set(fig,'PaperUnits','inches','PaperPositionMode','auto');
    box on;
    exportgraphics(fig, '1D3_histogram_dFdz.pdf', 'ContentType', 'vector', 'Resolution', 600);
    close(fig);
end

%%
% ---------------------------------
% 7. U(x) Plot
% ---------------------------------
U_file = dir('1D3_U_data.mat');
if ~isempty(U_file)
    S = load(U_file(1).name); % X, U
    fig = figure('Units','inches','Position',[0 0 4.3 2]);
    plot(S.X, S.U, 'Color', [0 0.45 0.74], 'LineWidth', 1.5);
    xlabel('x'); ylabel('u');
    set(gca,'FontName','Times New Roman');
    grid off; box on
    set(fig,'PaperUnits','inches','PaperPositionMode','auto');
    box on;
    ylim([-0.25, 0.25])
    exportgraphics(fig, '1D3_U.pdf', 'ContentType', 'vector', 'Resolution', 600);
    close(fig);
else
    warning('U(x) file not found.');
end
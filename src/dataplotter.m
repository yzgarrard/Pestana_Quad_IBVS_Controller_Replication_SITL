clear; close all; clc;

data = readtable('debug_data.csv'); % Read the data
data.time(:) = data.time(:) - data.time(1);

data.f_v_ref(:) = 0.5;

figure;

plot(data.time, data.x_bb, '-+',...
    data.time, data.y_bb, '-o',...
    data.time, data.w_bb, '-*',...
    data.time, data.h_bb, '-.',...
    data.time, data.v_xr.*100, '-^',...
    data.time, data.v_yr.*100, '->',...
    data.time, data.v_zr.*100, '-<',...
    data.time, data.yawrate.*10,  '-p',...
    data.time, data.f_u*100, '-x',...
    data.time, data.f_v*100, '-s',...
    data.time, data.f_delta*10, '-d',...
    data.time, data.delta_f_u_psi, '-^',...
    data.time, data.delta_f_u_y, '->',...
    data.time, data.delta_f_v_z, '-^',...
    data.time, data.delta_f_delta_x, '-p',...
    data.time, data.delta_x_tme, '-h',...
    data.time, data.delta_y_tme, '-+',...
    data.time, data.delta_z_tme, '-o',...
    data.time, data.delta_psi_tme, '-*',...
    data.time, data.prev_delta_x_tme, '-.',...
    data.time, data.prev_delta_y_tme, '-x',...
    data.time, data.prev_delta_z_tme, '-s',...
    data.time, data.prev_delta_psi_tme, '-d',...
    data.time, data.pitch_angle*100)

legend('x_{bb}',...
    'y_{bb}',...
    'w_{bb}',...
    'h_{bb}',...
    'v_{xr}*100',...
    'v_{yr}*100',...
    'v_{zr}*100',...
    'v_{yawrate}',...
    'f_u*100',...
    'f_v*100',...
    'f_{\Delta}*10',...
    '\Delta f_{u\psi}',...
    '\Delta f_{uy}',...
    '\Delta f_{vz}',...
    '\Delta f_{\Delta x}',...
    '\Delta x_{tme}',...
    '\Delta y_{tme}',...
    '\Delta z_{tme}',...
    '\Delta \psi_{tme}',...
    'prev \Delta_{xtme}',...
    'prev \Delta_{ytme}',...
    'prev \Delta_{zt,e}',...
    'prev \Delta_{\psi tme}',...
    'pitch angle*100')
% 
% figure;
% 
% plot(data.time, data.x_bb, '-+',...
%     data.time, data.y_bb, '-o',...
%     data.time, data.w_bb, '-*',...
%     data.time, data.h_bb, '-.',...
%     data.time, data.v_xr.*100, '-^',...
%     data.time, data.v_yr.*100, '->',...
%     data.time, data.v_zr.*100, '-<',...
%     data.time, data.yawrate.*100,  '-p',...
%     data.time, data.roll_angle.*100, '-^',...
%     data.time, data.pitch_angle.*100, '->',...
%     data.time, data.yaw_angle.*10, '-<',...
%     data.time, data.delta_f_delta_x.*100, '-p')
% 
% legend('x_{bb}',...
%     'y_{bb}',...
%     'w_{bb}',...
%     'h_{bb}',...
%     'v_{xr} * 100',...
%     'v_{yr} * 100',...
%     'v_{zr} * 100',...
%     'v_{yawrate} * 100',...
%     'roll angle * 100',...
%     'pitch angle * 100',...
%     'yaw angle * 10',...
%     'delta_{f\Delta_x} * 100')

% For f_delta plots
figure
plot(data.time, data.f_delta, data.time, data.f_delta_ref)
legend('f_\Delta measured', 'f_\Delta desired')
grid
title('f_\Delta measured vs f_\Delta desired for static target')
ylabel('f_\Delta')
xlabel('Time (s)')

% For f_v plots
figure
plot(data.time, data.f_v, data.time, data.f_v_ref)
legend('f_v measured', 'f_v desired')
grid
title('f_v measured vs f_v desired for static target')
ylabel('f_v')
xlabel('Time (s)')

% For f_x plots
figure
plot(data.time, data.vx_measured, data.time, data.v_xr)
legend('v_x measured', 'v_x desired')
grid
title('v_x measured vs v_x desired for static target')
ylabel('v_x')
xlabel('Time (s)')

% For f_z plots
figure
plot(data.time, data.vz_measured, data.time, data.v_zr)
legend('v_z measured', 'v_z desired')
grid
title('v_z measured vs v_z desired for static target')
ylabel('v_z')
xlabel('Time (s)')
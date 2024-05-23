clear
close all
clc

X0 = [1 -2 5]';
X_DFP = DFP(X0, 1e-10);
X_BFGS = BFGS(X0, 1e-10);
X_Simplex = Simplex(X0, 1e-10);
X_ConjugateGradiant = ConjugateGradiant(X0, 1e-10);

%% Functions
function Cost_Value = Cost(X) % Insert Cost Function Here
x1 = X(1); x2 = X(2); x3 = X(3);

Cost_Value = (1 - x1)^2 + (1-x2)^2+ 50*(x2-x1^2)^2 + 50*(x3-x2^2)^2;
end
function Jacibian_Value = Jacibian(X) % Insert Jacobian of the Cost Function Here
x1 = X(1); x2 = X(2); x3 = X(3);

Jacibian_Value = [ -2*(1 - x1)-200*(x2 - x1^2)*x1 ;
    -2*(1 - x2)+100*(x2 - x1^2)-200*(x3-x2^2)*x2 ;
    100*(x3 - x2^2)];
end
function X_Optimal = DFP(X0, accuracy) % DFP Method
L = length(X0);
H = eye(L);
X_Optimal = [];
residual = 1;

while residual > accuracy
    C = Jacibian(X0);
    D = -H*C;
    alpha = GoldenSection(0, 100, 0, 0, accuracy, X0, D);
    X1 = X0 + alpha*D;
    P = alpha*D;
    C1 = Jacibian(X1);
    Q = C1 - C;
    H1 = H + (P*P')/(P'*Q) - (H*Q*Q'*H)/(Q'*H*Q);
    
    H = H1;
    X0 = X1;
    X_Optimal = [X_Optimal; X0'];
    LL = length(X_Optimal(:,1));
    if LL > 1
        residual = ((X_Optimal(LL,:) - X_Optimal(LL - 1,:))*(X_Optimal(LL,:) - X_Optimal(LL - 1,:))')^0.5;
    end
end
end
function X_Optimal = BFGS(X0, accuracy) % BFGS Method
L = length(X0);
B = eye(L);
X_Optimal = [];
residual = 1;

while residual > accuracy
    C = Jacibian(X0);
    D = -B\C;
    alpha = GoldenSection(0, 100, 0, 0, accuracy, X0, D);
    X1 = X0 + alpha*D;
    P = alpha*D;
    C1 = Jacibian(X1);
    Q = C1 - C;
    B1 = B + (Q*Q')/(Q'*P) - (B*P*P'*B)/(P'*B*P);
    
    B = B1;
    X0 = X1;
    X_Optimal = [X_Optimal; X0'];
    LL = length(X_Optimal(:,1));
    if LL > 1
        residual = ((X_Optimal(LL,:) - X_Optimal(LL - 1,:))*(X_Optimal(LL,:) - X_Optimal(LL - 1,:))')^0.5;
    end
end
end
function X_Optimal = Simplex(X0, accuracy) % Simplex Method
a_r = 1; a_e = 3; a_c = -0.5; delta = 0.5;
X_Optimal = [];
residual = 1;

L = length(X0);
B = [zeros(L,1) eye(L)];

for i = 1:(L + 1)
    Cost_Value(1,i) = Cost(B(:,i));
end

while residual > accuracy
    Max_Cost_Point = find(Cost_Value==max(Cost_Value));
    worst_Point = B(:,Max_Cost_Point);
    Other_Points = B;
    Other_Points(:,Max_Cost_Point) = [];
    Other_Costs = Cost_Value;
    Other_Costs(:,Max_Cost_Point) = [];
    Center_Point = Other_Points*ones(L,1)/L;
    min_Cost = min(Other_Costs);
    max_Cost = max(Other_Costs);
    
    x_Reflection = (1 + a_r)*Center_Point - a_r*worst_Point; % Reflection
    Cost_Reflection = Cost(x_Reflection);
    B(:,Max_Cost_Point) = x_Reflection;
    Cost_Value(:,Max_Cost_Point) = Cost_Reflection;
    
    if Cost_Reflection >= min_Cost && Cost_Reflection < max_Cost % Keep Changes
        % Do Nothing
    elseif Cost_Reflection < min_Cost % Expantion
        x_Expantion = (1 + a_e)*Center_Point - a_e*worst_Point;
        Cost_Expantion = Cost(x_Expantion);
        if Cost_Expantion < Cost_Reflection % Keep Changes
            B(:,Max_Cost_Point) = x_Expantion;
            Cost_Value(:,Max_Cost_Point) = Cost_Expantion;
        end
    elseif Cost_Reflection >= max_Cost % Contraction
        x_Contraction = (1 + a_c)*Center_Point - a_c*worst_Point;
        Cost_Contraction = Cost(x_Contraction);
        if Cost_Contraction < Cost_Reflection % Keep Changes
            B(:,Max_Cost_Point) = x_Contraction;
            Cost_Value(:,Max_Cost_Point) = Cost_Contraction;
        else % Shrinking
            Min_Cost_Point = find(Cost_Value==min(Cost_Value));
            Best_Point = B(:,Min_Cost_Point);
            for j = 1:L
                B(:,j) = Best_Point + delta*(B(:,j) - Best_Point);
                Cost_Value(1,j) = Cost(B(:,j));
            end
        end
    end
    Min_Cost_Point = find(Cost_Value==min(Cost_Value));
    Max_Cost_Point = find(Cost_Value==max(Cost_Value));
    X_Optimal = [X_Optimal; B(:,Min_Cost_Point)'];
    residual = ((B(:,Min_Cost_Point)' - B(:,Max_Cost_Point)')*(B(:,Min_Cost_Point)' - B(:,Max_Cost_Point)')')^0.5;
end
end
function x_Optimal = GoldenSection(xl, xu, xm, iteration_type, accuracy, X0, dX) % Line Search: Golden Section
a = (5^0.5 - 1)/2;
residual = (xu - xl);

while residual > accuracy
    if iteration_type == 0 % first iteration
        xb = xl + a*(xu - xl);
        xa = xl + (1 - a)*(xu - xl);
    elseif iteration_type == 1 % xu is deleted
        xb = xm;
        xa = xl + (1 - a)*(xu - xl);
    elseif iteration_type == 2 % xu is deleted
        xb = xl + a*(xu - xl);
        xa = xm;
    end
    
    fa = Cost(X0 + xa*dX);
    fb = Cost(X0 + xb*dX);
    
    if fa >= fb
        xl = xa;
        xm = xb;
        iteration_type = 2;
    else
        xu = xb;
        xm = xa;
        iteration_type = 1;
    end
    residual = (xu - xl);
end
x_Optimal = xm;
end
function x_Optimal = Quadratic(xl, xu, xm, iteration_type, accuracy, X0, dX) % Line Search: Quadratic
residual = (xu - xl);

while residual > accuracy
    if iteration_type == 0 % first iteration
        xi = (xl + xu)/2;
        iteration_type = 1;
    end
    
    fl = Cost(X0 + xl*dX);
    fu = Cost(X0 + xu*dX);
    fi = Cost(X0 + xi*dX);
    a2 = (1/(xu - xi))*(((fu - fi)/(xu - xi)) - ((fi - fl)/(xi - xl)));
    a1 = ((fi - fl)/(xi - xl)) - (a2*(xi + xl));
    xm = -a1/(2*a2);    
    fm = Cost(X0 + xm*dX);
    
    if xm >= xi
        xl = xi;
        xi = xm;
    else
        xu = xi;
        xi = xm;
    end
    residual = (xu - xl);
end
x_Optimal = xm;
end
function X_Optimal = ConjugateGradiant(X0,accuracy)
X_Optimal = [];
residual =1;

while residual>accuracy
    C1 = Jacibian(X0);
    D1 = -C1;
    alpha = GoldenSection(0, 100, 0, 0, accuracy, X0, D1);%Compute Lambda
    X1 = X0+alpha.*D1; %Update X convert S to Row
    X0 = X1;
    C = C1;
    D = D1;
    X_Optimal = [X_Optimal; X0'];
    LL = length(X_Optimal(:,1));
    if LL > 1
        residual = ((X_Optimal(LL,:) - X_Optimal(LL - 1,:))*(X_Optimal(LL,:) - X_Optimal(LL - 1,:))')^0.5;
    end       
end
end

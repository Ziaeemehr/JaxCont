function out = lorenz84
% Independent extended Lorenz-84 transcription; parameters ordered F,T.
out = {@init,@rhs,@jac,@jacp,@hess,[],@third,[],[]};
end
function [t,y,o] = init
 t=[0 10]; y=[3;0;0;0]; o=odeset;
end
function f=rhs(~,s,F,T)
X=s(1);Y=s(2);Z=s(3);U=s(4);
f=[-Y^2-Z^2-.25*X+.25*F-.987*U^2; X*Y-X*Z-Y+.25; X*Y+X*Z-Z; -1.04*U+.987*U*X+T];
end
function J=jac(~,s,~,~)
X=s(1);Y=s(2);Z=s(3);U=s(4);
J=[-.25,-2*Y,-2*Z,-1.974*U;Y-Z,X-1,-X,0;Y+Z,X,X-1,0;.987*U,0,0,-1.04+.987*X];
end
function J=jacp(~,~,~,~)
J=[.25,0;0,0;0,0;0,1];
end
function H=hess(~,~,~,~)
H=zeros(4,4,4); H(1,2,2)=-2;H(1,3,3)=-2;H(1,4,4)=-1.974;
H(2,1,2)=1;H(2,2,1)=1;H(2,1,3)=-1;H(2,3,1)=-1;
H(3,1,2)=1;H(3,2,1)=1;H(3,1,3)=1;H(3,3,1)=1;
H(4,1,4)=.987;H(4,4,1)=.987;
end
function D=third(~,~,~,~)
D=zeros(4,4,4,4);
end

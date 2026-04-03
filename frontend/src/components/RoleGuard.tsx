import { Navigate, useLocation } from "react-router-dom";
import { hasRouteAccess, readRoleFromStorage } from "../authz";

interface Props {
  children: JSX.Element;
}

export function RoleGuard({ children }: Props) {
  const location = useLocation();
  const role = readRoleFromStorage();
  if (!hasRouteAccess(role, location.pathname)) {
    return <Navigate to="/apps/cbam" replace />;
  }
  return children;
}

import Alert from "@mui/material/Alert";
import Typography from "@mui/material/Typography";
import type { ReactNode } from "react";

export interface ConnectorIncident {
  connector_id: string;
  app_id: string;
  message: string;
}

export function ShellConnectorIncidentsAlert(props: {
  incidents: ConnectorIncident[];
  adminLink?: ReactNode;
}) {
  if (!props.incidents?.length) return null;
  return (
    <Alert severity="warning" role="status" sx={{ borderRadius: 0 }}>
      <Typography variant="body2">
        Connector incidents:{" "}
        {props.incidents.map((i) => (
          <span key={`${i.connector_id}-${i.app_id}`}>
            <strong>{i.connector_id}</strong> ({i.app_id}) — {i.message}{" "}
          </span>
        ))}
      </Typography>
      {props.adminLink}
    </Alert>
  );
}

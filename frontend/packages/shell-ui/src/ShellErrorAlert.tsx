import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

export interface ErrorEnvelope {
  title: string;
  message: string;
  details?: string[];
}

export interface ShellErrorAlertProps {
  envelope: ErrorEnvelope;
  severity?: "error" | "warning" | "info";
}

export function ShellErrorAlert({ envelope, severity = "error" }: ShellErrorAlertProps) {
  return (
    <Alert severity={severity}>
      <AlertTitle>{envelope.title}</AlertTitle>
      <Typography variant="body2">{envelope.message}</Typography>
      {envelope.details && envelope.details.length > 0 && (
        <Box component="ul" sx={{ pl: 2, mb: 0, mt: 1 }}>
          {envelope.details.map((d) => (
            <li key={d}>
              <Typography variant="caption" component="span">
                {d}
              </Typography>
            </li>
          ))}
        </Box>
      )}
    </Alert>
  );
}

/** Build envelope from API `error_envelope` or legacy string lists */
export function errorEnvelopeFromApi(
  envelope: ErrorEnvelope | null | undefined,
  errors?: string[] | null
): ErrorEnvelope | null {
  if (envelope?.message) return envelope;
  const list = errors?.filter(Boolean) ?? [];
  if (!list.length) return null;
  return {
    title: "Run encountered errors",
    message: list[0]!,
    details: list.slice(1, 12)
  };
}

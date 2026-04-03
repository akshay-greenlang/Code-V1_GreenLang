import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

export function AdminPage() {
  return (
    <Stack spacing={2}>
      <Typography variant="h5">Admin Console</Typography>
      <Card variant="outlined">
        <CardContent>
          <Typography variant="body2">
            Role-aware administration, release train visibility, and connector health monitoring.
          </Typography>
        </CardContent>
      </Card>
    </Stack>
  );
}

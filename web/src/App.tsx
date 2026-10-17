import { Routes, Route } from "react-router-dom";
import { StatusProvider } from "./lib/status";
import { ConversationsProvider } from "./lib/conversations";
import { SettingsProvider } from "./lib/settings";
import { Shell } from "./components/Shell";
import { ChatPage } from "./pages/ChatPage";
import { ModelsPage } from "./pages/ModelsPage";
import { MonitorPage } from "./pages/MonitorPage";
import { SettingsPage } from "./pages/SettingsPage";

export default function App() {
  return (
    <SettingsProvider>
      <StatusProvider>
        <ConversationsProvider>
          <Shell />
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/monitor" element={<MonitorPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<ChatPage />} />
          </Routes>
        </ConversationsProvider>
      </StatusProvider>
    </SettingsProvider>
  );
}

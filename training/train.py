import torch.optim as optim

# optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss functions
adversarial_loss = nn.BCELoss()
content_loss = nn.L1Loss()

num_epochs = 20
for epoch in range(num_epochs):
    for blurred, sharp in train_loader:
        blurred, sharp = blurred.cuda(), sharp.cuda()
    
        batch_size = sharp.shape[0]  # Get actual batch size
        d_out_shape = discriminator(sharp).shape[2:]  # Extract (15,15)
    
        real_labels = torch.ones((batch_size, 1, *d_out_shape)).cuda()
        fake_labels = torch.zeros((batch_size, 1, *d_out_shape)).cuda()
    
        # Train Discriminator
        fake_sharp = generator(blurred).detach()
        
        d_loss_real = adversarial_loss(discriminator(sharp), real_labels)
        d_loss_fake = adversarial_loss(discriminator(fake_sharp), fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
    
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
    
        # Train Generator
        g_loss = adversarial_loss(discriminator(generator(blurred)), real_labels) + content_loss(generator(blurred), sharp)
    
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    if (epoch + 1) % 5 == 0:
        torch.save(generator.state_dict(), f"DeblurGAN_epoch_{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")

print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
